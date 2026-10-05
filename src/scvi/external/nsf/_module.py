from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal, Poisson

from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data

from ._constants import NSF_REGISTRY_KEYS

if TYPE_CHECKING:
    from scvi._types import Tensor

logger = logging.getLogger(__name__)


class NSFModule(BaseModuleClass):
    r"""Module for Non-negative Spatial Factorization (NSF) :cite:p:`Townes2023`.

    Clean-room PyTorch reimplementation of the NSF model described in
    `arXiv:2110.06122 <https://arxiv.org/abs/2110.06122>`_ (the authors'
    TensorFlow-Probability implementation was not used). Requested in
    `scvi-tools issue #1859 <https://github.com/scvi-tools/scvi-tools/issues/1859>`_.

    Generative model
    ----------------
    For spots :math:`i = 1, \dots, N` at spatial coordinates :math:`x_i` and genes
    :math:`j = 1, \dots, G`, the model factorizes Poisson count data with
    non-negative spatial factors:

    .. math::

        f_l(\cdot) &\sim \mathcal{GP}\left(0, k_l(\cdot, \cdot)\right),
        \qquad l = 1, \dots, L \\
        F_{il} &= \exp\left(f_l(x_i)\right) \geq 0 \\
        W_{jl} &\geq 0 \\
        y_i &\sim \mathrm{Poisson}\left(F W^\top\right)_i

    where each factor :math:`l` has its own squared-exponential (RBF) kernel
    :math:`k_l(x, x') = \exp(-\lVert x - x' \rVert^2 / 2\ell_l^2)` over the
    (standardized) spatial coordinates, so factors are non-negative *and*
    spatially smooth. The zero-mean GP prior on the log scale anchors the
    factors' scale (median near 1), leaving gene-wise magnitude to the
    loadings :math:`W` (a half-normal prior scale is exposed via
    ``loadings_scale``; :math:`W` is point-estimated, MAP-style).

    Variational inference
    ---------------------
    A sparse variational GP with :math:`M` learnable inducing points
    :math:`Z` (Titsias 2009; Hensman et al. 2013) is used for scalability:
    :math:`q(u_l) = \mathcal{N}(m_l, \mathrm{diag}(s_l^2))` at the inducing
    locations, from which the posterior over each factor at any coordinate is
    obtained by GP conditioning. The negative ELBO minimized per minibatch is

    .. math::

        \mathcal{L} = \frac{1}{B} \sum_{i \in B} \mathbb{E}_{q}
        \left[-\log p(y_i \mid f_i, W)\right]
        + \frac{1}{N} \left[
            \sum_{l=1}^{L} \mathrm{KL}(q(u_l) \,\|\, p(u_l)) +
            \frac{1}{2\sigma_w^2} \lVert W \rVert_F^2
        \right]

    where the minibatch KL scaling follows the scvi-tools convention
    (``n_obs`` is injected into :meth:`loss` by the
    :class:`~scvi.train.TrainingPlan`). The likelihood expectation is estimated
    with reparameterized Monte Carlo samples of the factor posterior.

    Parameters
    ----------
    n_input
        Number of genes (:math:`G`).
    n_factors
        Number of non-negative spatial factors (:math:`L`).
    n_inducing_points
        Number of GP inducing points (:math:`M`). Defaults to
        ``min(100, n_spots)``.
    spatial_coords
        Full ``(n_spots, d)`` matrix of spatial coordinates. Used only at
        initialization to (1) standardize coordinates (the module stores the
        per-dimension mean/std as buffers) and (2) initialize the inducing
        locations by subsampling. It is *not* retained; per-minibatch
        coordinates arrive through the data loader.
    lengthscale_init
        Initial RBF lengthscale (on standardized coordinates). Lengthscales
        are learnable and initialized on a multi-scale ladder spanning
        ``[0.5, 2] x lengthscale_init`` across factors.
    loadings_scale
        Scale :math:`\sigma_w` of the half-normal prior (L2 penalty) on the
        non-negative loadings.
    n_mc_samples
        Number of reparameterized Monte Carlo samples used to estimate the
        Poisson likelihood expectation during training.
    jitter
        Initial diagonal jitter added to the inducing-point Gram matrix
        before Cholesky factorization (escalated on failure).
    """

    def __init__(
        self,
        n_input: int,
        n_factors: int = 10,
        n_inducing_points: int | None = None,
        spatial_coords: Tensor | None = None,
        lengthscale_init: float = 0.5,
        loadings_scale: float = 1.0,
        n_mc_samples: int = 1,
        jitter: float = 1e-4,
    ):
        super().__init__()
        if spatial_coords is None:
            raise ValueError("`spatial_coords` (n_spots, d) is required to initialize NSFModule.")
        if spatial_coords.ndim != 2 or spatial_coords.shape[0] < 2:
            raise ValueError(
                "`spatial_coords` must be a 2D tensor with at least two spots, got shape "
                f"{tuple(spatial_coords.shape)}."
            )

        self.n_input = n_input
        self.n_factors = n_factors
        self.n_mc_samples = n_mc_samples
        self.loadings_scale = loadings_scale
        self.n_spots = int(spatial_coords.shape[0])

        # Standardize coordinates so a single lengthscale scale is meaningful.
        coord_mean = spatial_coords.mean(dim=0)
        coord_std = spatial_coords.std(dim=0).clamp_min(1e-8)
        self.register_buffer("coord_mean", coord_mean)
        self.register_buffer("coord_std", coord_std)
        coords = (spatial_coords - coord_mean) / coord_std

        m = min(n_inducing_points or 100, self.n_spots)
        self.n_inducing_points = m
        # Deterministic subsample of the spots as initial inducing locations.
        generator = torch.Generator().manual_seed(0)
        inducing_idx = torch.randperm(coords.shape[0], generator=generator)[:m]
        self.inducing_locs = nn.Parameter(coords[inducing_idx].clone())

        # Variational distribution q(u_l) = N(m_l, diag(s_l^2)), one per factor.
        self.q_mu = nn.Parameter(torch.zeros(n_factors, m))
        self.q_scale_shift = nn.Parameter(torch.full((n_factors, m), -3.5))

        # Per-factor RBF lengthscales on a multi-scale ladder, learnable.
        if n_factors > 1:
            ladder = 0.5 + torch.arange(n_factors, dtype=torch.float32) * (1.5 / (n_factors - 1))
        else:
            ladder = torch.ones(1)
        target_ls = lengthscale_init * ladder
        # Inverse softplus so that softplus(raw) == target_ls.
        raw = self._inverse_softplus(target_ls.clamp_min(1e-3))
        self.log_lengthscale_raw = nn.Parameter(raw)

        # Unconstrained parameter for the non-negative loadings W = softplus(W_raw).
        generator = torch.Generator().manual_seed(0)
        self.W_raw = nn.Parameter(
            -1.0 + 0.1 * torch.randn(n_input, n_factors, generator=generator)
        )

        self.jitter = jitter

    @property
    def lengthscales(self) -> Tensor:
        """Per-factor RBF lengthscales ``(n_factors,)``."""
        return torch.nn.functional.softplus(self.log_lengthscale_raw) + 1e-2

    @property
    def loadings(self) -> Tensor:
        """Non-negative loadings ``W`` of shape ``(n_genes, n_factors)``."""
        return torch.nn.functional.softplus(self.W_raw)

    @staticmethod
    def _sq_dist(a: Tensor, b: Tensor) -> Tensor:
        """Squared euclidean distance matrix between rows of ``a`` and ``b``."""
        a2 = (a * a).sum(dim=1, keepdim=True)
        b2 = (b * b).sum(dim=1).unsqueeze(0)
        return (a2 + b2 - 2 * a @ b.T).clamp_min(0.0)

    @staticmethod
    def _inverse_softplus(y: Tensor) -> Tensor:
        return torch.log(torch.expm1(y))

    def _cholesky(self, k_zz: Tensor) -> Tensor:
        """Cholesky of ``k_zz`` with diagonal jitter, escalating on failure."""
        eye = torch.eye(k_zz.shape[-1], device=k_zz.device, dtype=k_zz.dtype)
        jitter = self.jitter
        for _ in range(5):
            try:
                return torch.linalg.cholesky(k_zz + jitter * eye)
            except RuntimeError:
                jitter = jitter * 10.0
        return torch.linalg.cholesky(k_zz + jitter * eye)

    def _normalize_coords(self, coords: Tensor) -> Tensor:
        return (coords - self.coord_mean) / self.coord_std

    def _inducing_gram(self) -> tuple[Tensor, Tensor]:
        """Prior Gram matrices at the inducing locations.

        Returns
        -------
        ``(k_zz, chol_zz)`` with shapes ``(n_factors, m, m)``.
        """
        d_zz = self._sq_dist(self.inducing_locs, self.inducing_locs)
        inv_two_ls_sq = 1.0 / (2.0 * self.lengthscales.square())  # (n_factors,)
        k_zz = torch.exp(-d_zz.unsqueeze(0) * inv_two_ls_sq.view(-1, 1, 1))
        return k_zz, self._cholesky(k_zz)

    def _gp_posterior(self, coords: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sparse-GP posterior over the log factors at ``coords``.

        Parameters
        ----------
        coords
            Raw (unnormalized) coordinates of shape ``(batch, d)``.

        Returns
        -------
        ``(mean, var, k_zz, chol_zz)`` where ``mean`` and ``var`` are the
        per-spot posterior mean and variance of each log factor, shapes
        ``(batch, n_factors)``; the Gram pieces are returned for the KL term.
        """
        coords = self._normalize_coords(coords)
        k_zz, chol_zz = self._inducing_gram()
        inv_two_ls_sq = 1.0 / (2.0 * self.lengthscales.square())

        d_zx = self._sq_dist(self.inducing_locs, coords)  # (m, batch)
        k_zx = torch.exp(-d_zx.unsqueeze(0) * inv_two_ls_sq.view(-1, 1, 1))  # (L, m, batch)

        # Posterior mean: K_zx^T K_zz^{-1} q_mu.
        alpha = torch.cholesky_solve(self.q_mu.unsqueeze(-1), chol_zz).squeeze(-1)  # (L, m)
        mean = torch.einsum("lmb,lm->bl", k_zx, alpha)

        # Posterior variance (diagonal only):
        # K_xx - K_xz K_zz^-1 K_zx + K_xz K_zz^-1 S K_zz^-1 K_zx with S = diag(q_scale^2).
        w = torch.cholesky_solve(k_zx, chol_zz)  # (L, m, batch)
        q_scale = torch.nn.functional.softplus(self.q_scale_shift) + 1e-3
        var = (
            1.0 - (k_zx * w).sum(dim=1) + (w.square() * q_scale.square().unsqueeze(-1)).sum(dim=1)
        )  # (L, batch)
        var = var.transpose(0, 1).clamp_min(1e-8)  # (batch, L)
        return mean, var, k_zz, chol_zz

    def _get_inference_input(self, tensors: dict[str, Tensor], **kwargs):
        return {
            "spatial_coords": tensors[NSF_REGISTRY_KEYS.SPATIAL_COORDS_KEY],
        }

    def _get_generative_input(
        self,
        tensors: dict[str, Tensor],
        inference_outputs: dict[str, Tensor],
        **kwargs,
    ):
        return {"f": inference_outputs["f"]}

    @auto_move_data
    def inference(
        self,
        spatial_coords: Tensor,
        n_samples: int | None = None,
    ) -> dict[str, Tensor | Normal]:
        r"""Run the variational posterior over the log spatial factors.

        The recognition model is the sparse GP itself: conditioning
        :math:`q(u)` on the minibatch coordinates yields the posterior over
        each spot's log factors (there is no amortizing encoder network).

        Parameters
        ----------
        spatial_coords
            Coordinates of the minibatch spots, shape ``(batch, d)``.
        n_samples
            Number of reparameterized samples of the log factors to draw.
            Defaults to ``n_mc_samples`` while training and to 1 otherwise.

        Returns
        -------
        Dictionary with

        ``qz``
            :class:`~torch.distributions.Normal` over log factors,
            shape ``(batch, n_factors)``.
        ``z``
            The first sample of the log factors, shape ``(batch, n_factors)``.
            (Kept for compatibility with
            :meth:`~scvi.model.base.VAEMixin.get_latent_representation`, which
            returns the log-space factors.)
        ``f``
            Sample(s) of the log factors, shape ``(n_samples, batch, n_factors)``.
        """
        if n_samples is None:
            n_samples = self.n_mc_samples if self.training else 1
        mean, var, _, _ = self._gp_posterior(spatial_coords)
        qz = Normal(mean, var.sqrt())
        # Reparameterized samples for the MC estimate of the likelihood term.
        f = qz.rsample((n_samples,))
        return {"qz": qz, "z": f[0], "f": f}

    @auto_move_data
    def generative(self, f: Tensor, **kwargs) -> dict[str, Poisson]:
        r"""Generate the Poisson likelihood parameters.

        Parameters
        ----------
        f
            Samples of the log spatial factors, shape ``(n_samples, batch, n_factors)``.

        Returns
        -------
        Dictionary with ``px``, a :class:`~torch.distributions.Poisson` over
        counts with rate :math:`\Lambda^s = \exp(f^s) W^\top` per sample.
        """
        w = self.loadings
        # Clamp the log factors as a safety valve against overflow/underflow.
        f = f.clamp(-20.0, 20.0)
        lam = torch.exp(f) @ w.T  # (n_samples, batch, n_genes)
        return {"px": Poisson(lam)}

    def loss(
        self,
        tensors: dict[str, Tensor],
        inference_outputs: dict[str, Tensor | Normal],
        generative_outputs: dict[str, Poisson],
        kl_weight: torch.Tensor | float = 1.0,
        n_obs: int | None = None,
    ) -> LossOutput:
        """Compute the minibatch negative ELBO (scaled per observation)."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        if n_obs is None:
            n_obs = self.n_spots
        n_obs = max(n_obs, 1)

        # Poisson reconstruction loss, per spot (averaged over MC samples).
        rec_loss = -generative_outputs["px"].log_prob(x).sum(dim=-1).mean(dim=0)  # (batch,)

        # KL[q(u) || p(u)] summed over factors (a global term).
        _, chol_zz = self._inducing_gram()
        q_scale = torch.nn.functional.softplus(self.q_scale_shift) + 1e-3
        qu = MultivariateNormal(self.q_mu, scale_tril=torch.diag_embed(q_scale))
        pu = MultivariateNormal(
            torch.zeros_like(self.q_mu),
            scale_tril=chol_zz,
        )
        kl_gp = torch.distributions.kl_divergence(qu, pu).sum()

        # Half-normal-style L2 penalty on the non-negative loadings (W is
        # point-estimated, so this is a MAP penalty rather than an exact
        # marginalization).
        loadings_penalty = 0.5 * (self.loadings / self.loadings_scale).square().sum()

        kl_global = kl_gp + loadings_penalty
        loss = rec_loss.mean() + kl_weight * kl_global / n_obs

        return LossOutput(
            loss=loss,
            reconstruction_loss=rec_loss,
            kl_local=torch.zeros_like(rec_loss),
            kl_global=kl_global,
            extra_metrics={"mean_lengthscale": self.lengthscales.mean().detach()},
        )

    @torch.inference_mode()
    def sample(
        self,
        tensors: dict[str, Tensor],
        n_samples: int = 1,
        max_poisson_rate: float = 1e8,
    ) -> Tensor:
        r"""Sample from the posterior predictive count distribution.

        Parameters
        ----------
        tensors
            Minibatch of data with counts and spatial coordinates.
        n_samples
            Number of posterior predictive samples per spot.
        max_poisson_rate
            Upper bound on the Poisson rate for numerical stability.

        Returns
        -------
        Tensor of shape ``(n_samples, batch, n_genes)``.
        """
        inference_outputs = self.inference(**self._get_inference_input(tensors), n_samples=1)
        px = self.generative(**self._get_generative_input(tensors, inference_outputs))["px"]
        rate = px.rate.clamp(max=max_poisson_rate)[0]  # (batch, n_genes)
        return Poisson(rate).sample((n_samples,))
