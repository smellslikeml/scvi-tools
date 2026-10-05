"""Tests for the NSF (non-negative spatial factorization) model.

These are CPU smoke tests: they prove that the model can be constructed,
registered, trained through the scvi-tools ``TrainingPlan`` contract, and that
its factors/loadings have the promised shapes, are finite and non-negative,
and that the ELBO improves. They do not prove recovery of spatial structure
(see the data-recovery validation plan in the model documentation).
"""

import numpy as np
import pytest
import torch
from anndata import AnnData

import scvi
from scvi.external import NSF
from scvi.external.nsf._constants import NSF_REGISTRY_KEYS


def _synthetic_spatial(n_spots=120, n_genes=30, n_factors=3, seed=0):
    """Synthetic spatial AnnData with smooth non-negative spatial factors.

    Log-factor fields are smooth trigonometric functions of the coordinates and
    genes are assigned block-wise to factors, so the ground-truth structure is
    spatially smooth and non-negative.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, size=(n_spots, 2)).astype(np.float32)
    fields = [
        np.sin(coords[:, 0] / 3.0),
        np.cos(coords[:, 1] / 3.0),
        np.sin((coords[:, 0] + coords[:, 1]) / 5.0),
    ]
    factors = np.exp(0.5 * np.stack(fields, axis=1)[:, :n_factors])
    loadings = 0.2 * rng.uniform(0.1, 1.0, size=(n_genes, n_factors))
    block = n_genes // n_factors
    for b in range(n_factors):
        loadings[b * block : (b + 1) * block, b] += rng.uniform(0.5, 1.5, size=block)
    lam = factors @ loadings.T
    counts = rng.poisson(lam).astype(np.float32)
    return AnnData(X=counts, obsm={"spatial": coords})


def test_nsf_setup_anndata_registers_spatial_coords():
    scvi.settings.seed = 0
    adata = _synthetic_spatial()
    NSF.setup_anndata(adata, spatial_key="spatial")
    model = NSF(adata, n_factors=2, n_inducing_points=10)
    registered = np.asarray(
        model.adata_manager.get_from_registry(NSF_REGISTRY_KEYS.SPATIAL_COORDS_KEY)
    )
    np.testing.assert_array_equal(registered, adata.obsm["spatial"])


def test_nsf_setup_missing_spatial_key_raises():
    adata = _synthetic_spatial()
    del adata.obsm["spatial"]
    with pytest.raises(KeyError):
        NSF.setup_anndata(adata, spatial_key="spatial")


def test_nsf_non_finite_coords_raise():
    adata = _synthetic_spatial()
    adata.obsm["spatial"][0, 0] = np.nan
    NSF.setup_anndata(adata, spatial_key="spatial")
    with pytest.raises(ValueError, match="non-finite"):
        NSF(adata)


def test_nsf_smoke_train_and_getters():
    """Construct, train on CPU, and check factor/loading shapes and finiteness."""
    scvi.settings.seed = 0
    adata = _synthetic_spatial()
    NSF.setup_anndata(adata, spatial_key="spatial")
    model = NSF(adata, n_factors=3, n_inducing_points=15)
    model.train(max_epochs=10, accelerator="cpu", batch_size=64)

    factors = model.get_factors()
    loadings = model.get_loadings()
    assert factors.shape == (adata.n_obs, 3)
    assert loadings.shape == (3, adata.n_vars)
    assert list(factors.columns) == [f"NSF_factor_{i}" for i in range(3)]
    assert np.all(np.isfinite(factors.values))
    assert np.all(np.isfinite(loadings.values))
    assert np.all(factors.values >= 0)
    assert np.all(loadings.values >= 0)

    # The VAEMixin latent representation is the log-space factor posterior mean.
    log_factors = model.get_latent_representation()
    assert log_factors.shape == (adata.n_obs, 3)
    assert np.all(np.isfinite(log_factors))

    # The logged elbo_train is the negative ELBO (reconstruction plus KL terms),
    # so it decreases as the model fits. Per-epoch values are noisy (minibatch
    # MC), so compare the last epochs against the first ones.
    elbo = model.history_["elbo_train"].astype(float).to_numpy().ravel()
    assert np.isfinite(elbo).all()
    assert elbo[-1] < elbo[0]
    assert elbo[-2:].mean() < elbo[:2].mean()


def test_nsf_get_factors_numpy_and_samples():
    scvi.settings.seed = 0
    adata = _synthetic_spatial(n_spots=60, n_genes=20)
    NSF.setup_anndata(adata, spatial_key="spatial")
    model = NSF(adata, n_factors=2, n_inducing_points=10)
    model.train(max_epochs=1, accelerator="cpu", batch_size=60)

    factors_np = model.get_factors(return_numpy=True)
    assert isinstance(factors_np, np.ndarray)
    assert factors_np.shape == (60, 2)
    assert np.all(factors_np >= 0)

    factors_sampled = model.get_factors(use_mean=False, n_samples=5, return_numpy=True)
    assert factors_sampled.shape == (60, 2)
    assert np.all(np.isfinite(factors_sampled))

    loadings_np = model.get_loadings(return_numpy=True)
    assert loadings_np.shape == (2, 20)
    assert np.all(loadings_np >= 0)


def test_nsf_module_elbo_decreases():
    """Direct gradient descent on the module loss decreases the negative ELBO."""
    scvi.settings.seed = 0
    adata = _synthetic_spatial()
    NSF.setup_anndata(adata, spatial_key="spatial")
    model = NSF(adata, n_factors=3, n_inducing_points=15)
    module = model.module
    dataloader = model._make_data_loader(adata, batch_size=64, shuffle=True)
    tensors = next(iter(dataloader))

    torch.manual_seed(0)
    module.train()
    optimizer = torch.optim.Adam(module.parameters(), lr=5e-2)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        _, _, loss_output = module(tensors, loss_kwargs={"kl_weight": 1.0, "n_obs": adata.n_obs})
        loss_output.loss.backward()
        optimizer.step()
        losses.append(loss_output.loss.item())

    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]


def test_nsf_module_posterior_predictive_sample():
    scvi.settings.seed = 0
    adata = _synthetic_spatial(n_spots=40, n_genes=15)
    NSF.setup_anndata(adata, spatial_key="spatial")
    model = NSF(adata, n_factors=2, n_inducing_points=10)
    dataloader = model._make_data_loader(adata, batch_size=40, shuffle=False)
    tensors = next(iter(dataloader))

    y = model.module.sample(tensors, n_samples=3).numpy()
    assert y.shape == (3, 40, 15)
    assert np.all(y >= 0)
    assert np.all(y == np.floor(y))
