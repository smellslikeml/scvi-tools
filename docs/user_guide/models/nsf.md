# NSF

**NSF** {cite:p}`Townes2023` (Python class {class}`~scvi.external.NSF`) is a
non-negative spatial factorization model for spatial transcriptomics. It
factorizes gene counts as $Y \sim \mathrm{Poisson}(F W^\top)$, where the spot
factors $F$ are non-negative (an $\exp$ link of Gaussian latent fields) and
spatially smooth: each log-factor is a zero-mean Gaussian process over the
spot coordinates with its own learnable squared-exponential lengthscale.
Inference is variational, with inducing points for scalability.

The advantages of NSF are:

-   Factors are non-negative *and* spatially smooth, which makes them
    directly interpretable as spatial gene-expression programs (niches,
    layers, domains).
-   Per-factor lengthscales are learned, so different factors can capture
    spatial structure at different scales.
-   The Poisson likelihood models counts directly, without a preliminary
    normalization step.

The limitations of NSF include:

-   It requires spot coordinates in `adata.obsm` (e.g. `spatial` for Visium).
-   The current implementation is a clean-room PyTorch reimplementation from
    the paper; it has not been numerically validated against the authors'
    reference implementation.
-   The hybrid NSFH variant (spatial plus non-spatial factors) and the
    real-valued RSF variant are not implemented.

## Preliminaries

NSF takes as input a spot-by-gene count matrix $X$ with $N$ spots and $G$
genes, together with the $(N, d)$ matrix of spatial coordinates. The counts
must be raw (unnormalized) integer counts; the coordinates are any key in
`adata.obsm` (e.g. `spatial`).

```python
import scanpy as sc
import scvi

adata = sc.read_h5ad("spatial_data.h5ad")
scvi.external.NSF.setup_anndata(adata, spatial_key="spatial")
```

## Getting started

```python
model = scvi.external.NSF(adata, n_factors=10)
model.train()

factors = model.get_factors()  # (n_spots, n_factors) DataFrame
loadings = model.get_loadings()  # (n_factors, n_genes) DataFrame

adata.obsm["X_nsf"] = factors.values
```

Both `factors` and `loadings` are non-negative by construction. Spatially
coherent factors can be visualized by plotting the columns of
`adata.obsm["X_nsf"]` on the spot coordinates (e.g. with
`squidpy.pl.spatial_scatter`), and the gene weights of each factor are
read off the corresponding row of `loadings`.

## Model architecture

$Y \sim \mathrm{Poisson}(\Lambda)$ with $\Lambda = F W^\top$ where $W \geq 0$
are non-negative gene loadings and $F = \exp(f)$ are non-negative spot factors
whose log values $f_l(\cdot)$ have zero-mean GP priors over the (standardized)
coordinates. A sparse variational GP approximation with learnable inducing
points is used for inference; the ELBO combines the Poisson reconstruction
term with the KL divergence between the variational GP posterior and the GP
prior, plus a half-normal prior (penalty) on the loadings. See
{class}`~scvi.external.nsf.NSFModule` for details.
