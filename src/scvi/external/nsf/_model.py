from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField,
    ObsmField,
)
from scvi.model.base import (
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
)
from scvi.train._config import merge_kwargs
from scvi.utils import setup_anndata_dsp

from ._constants import NSF_REGISTRY_KEYS
from ._module import NSFModule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

logger = logging.getLogger(__name__)


class NSF(UnsupervisedTrainingMixin, VAEMixin, BaseModelClass):
    r"""Non-negative Spatial Factorization :cite:p:`Townes2023`.

    NSF factorizes spatial transcriptomics count data into a small number of
    non-negative factors whose spatial patterns are governed by
    Gaussian-process priors over the spot coordinates, fit by (sparse)
    variational inference. Counts are modeled as
    :math:`Y \sim \mathrm{Poisson}(F W^\top)` with non-negative spot
    factors :math:`F = \exp(f)`, where each log factor :math:`f_l(\cdot)` is
    a zero-mean GP with its own learnable squared-exponential lengthscale,
    approximated with inducing points.

    This is a clean-room PyTorch reimplementation from the paper
    (`arXiv:2110.06122 <https://arxiv.org/abs/2110.06122>`_); the authors'
    reference implementation was not used. Requested in
    `scvi-tools issue #1859 <https://github.com/scvi-tools/scvi-tools/issues/1859>`_.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.external.NSF.setup_anndata`.
    n_factors
        Number of non-negative spatial factors.
    n_inducing_points
        Number of GP inducing points. Defaults to ``min(100, n_spots)``.
    lengthscale_init
        Initial spatial lengthscale (on standardized coordinates).
    loadings_scale
        Half-normal prior scale (L2 penalty) on the non-negative loadings.
    **model_kwargs
        Keyword args for :class:`~scvi.external.nsf.NSFModule`.

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.external.NSF.setup_anndata(adata, spatial_key="spatial")
    >>> nsf = scvi.external.NSF(adata)
    >>> nsf.train()
    >>> adata.obsm["X_nsf"] = nsf.get_factors()

    Notes
    -----
    See the module :class:`~scvi.external.nsf.NSFModule` for the generative
    model and ELBO. The hybrid NSFH (spatial plus non-spatial factors) and
    real-valued RSF variants of the paper are not implemented.
    """

    _module_cls = NSFModule

    def __init__(
        self,
        adata: AnnData,
        n_factors: int = 10,
        n_inducing_points: int | None = None,
        lengthscale_init: float = 0.5,
        loadings_scale: float = 1.0,
        **model_kwargs,
    ):
        super().__init__(adata)

        spatial_coords = self.adata_manager.get_from_registry(NSF_REGISTRY_KEYS.SPATIAL_COORDS_KEY)
        if hasattr(spatial_coords, "to_numpy"):
            spatial_coords = spatial_coords.to_numpy()
        spatial_coords = np.asarray(spatial_coords, dtype=np.float32)
        if spatial_coords.shape[0] != adata.n_obs:
            raise ValueError(
                f"Spatial coordinates have {spatial_coords.shape[0]} rows but the AnnData has "
                f"{adata.n_obs} spots."
            )
        if not np.isfinite(spatial_coords).all():
            raise ValueError("Spatial coordinates contain non-finite values.")

        if n_inducing_points is None:
            n_inducing_points = min(100, adata.n_obs)

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_factors=n_factors,
            n_inducing_points=n_inducing_points,
            spatial_coords=torch.from_numpy(spatial_coords),
            lengthscale_init=lengthscale_init,
            loadings_scale=loadings_scale,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"NSF Model with the following params: \nn_factors: {n_factors}, "
            f"n_inducing_points: {self.module.n_inducing_points}, "
            f"lengthscale_init: {lengthscale_init}, loadings_scale: {loadings_scale}"
        )
        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: int | None = None,
        lr: float = 1e-2,
        batch_size: int = 128,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        """Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. Defaults to a heuristic based on
            the dataset size.
        lr
            Learning rate for optimization. NSF has no encoder network (all
            parameters are global GP/loadings parameters), so a larger default
            than the scvi-tools convention of ``1e-3`` is used.
        batch_size
            Minibatch size to use during training.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments
            passed to this method will overwrite values present in
            ``plan_kwargs``, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        plan_kwargs = merge_kwargs(None, plan_kwargs, name="plan")
        plan_kwargs.update({"lr": lr})
        super().train(
            max_epochs=max_epochs,
            batch_size=batch_size,
            plan_kwargs=plan_kwargs,
            **kwargs,
        )

    @torch.inference_mode()
    def get_factors(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        use_mean: bool = True,
        n_samples: int = 10,
        return_numpy: bool = False,
    ) -> pd.DataFrame | np.ndarray:
        r"""Return the non-negative spatial factors.

        Factors are evaluated at each spot's spatial coordinates from the
        sparse-GP posterior; non-negativity comes from the ``exp`` link.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to the initial AnnData. If
            ``None``, defaults to the object used to initialize the model.
        indices
            Indices of spots in ``adata`` to use. If ``None``, all spots are used.
        batch_size
            Minibatch size for the forward pass. If ``None``, defaults to
            ``scvi.settings.batch_size``.
        use_mean
            If ``True`` (default), return the posterior mean of each factor,
            :math:`\exp(\mu + \sigma^2 / 2)` for the log-normal factor. If
            ``False``, average ``n_samples`` posterior draws of
            :math:`\exp(f)`.
        n_samples
            Number of posterior samples when ``use_mean`` is ``False``.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a
            :class:`~pandas.DataFrame`.

        Returns
        -------
        Factors of shape ``(n_spots, n_factors)``. DataFrame has spot names as
        index and ``NSF_factor_{i}`` columns.
        """
        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        factors = []
        for tensors in dataloader:
            outputs = self.module.inference(**self.module._get_inference_input(tensors))
            qz = outputs["qz"]
            if use_mean:
                # Mean of the log-normal factor exp(f), f ~ N(mu, sigma^2).
                spot_factors = torch.exp(qz.loc + 0.5 * qz.scale.square())
            else:
                spot_factors = torch.exp(qz.sample([n_samples])).mean(dim=0)
            factors.append(spot_factors.cpu())
        factors = torch.cat(factors).numpy()

        if return_numpy:
            return factors
        return pd.DataFrame(
            factors,
            index=adata.obs_names,
            columns=[f"NSF_factor_{i}" for i in range(factors.shape[1])],
        )

    @torch.inference_mode()
    def get_loadings(self, return_numpy: bool = False) -> pd.DataFrame | np.ndarray:
        """Return the non-negative gene loadings.

        Returns
        -------
        Loadings of shape ``(n_factors, n_genes)``. DataFrame has
        ``NSF_factor_{i}`` index and gene names as columns.
        """
        w = self.module.loadings
        w = w.T.detach().cpu().numpy()
        if return_numpy:
            return w
        return pd.DataFrame(
            w,
            index=[f"NSF_factor_{i}" for i in range(w.shape[0])],
            columns=self.adata.var_names,
        )

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        spatial_key: str = "spatial",
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        spatial_key
            Key in ``adata.obsm`` containing the ``(n_spots, d)`` spatial
            coordinates, e.g. ``"spatial"`` or ``"X_spatial"`` for Visium
            objects read with :func:`~scanpy.read_visium`.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            ObsmField(NSF_REGISTRY_KEYS.SPATIAL_COORDS_KEY, spatial_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
