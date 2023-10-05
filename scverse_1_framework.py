# %% [markdown]
# # AnnData

# %%
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix  # compressed sparse row mtx

# %%
# 100 cells x 2000 genes
counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
adata = ad.AnnData(counts)
adata

# %%
# access obs_names & var_names
adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
print(adata.obs_names[:10])

# %%
ct = np.random.choice(["B", "T", "Monocyte"], size=(adata.n_obs,))
adata.obs["cell_type"] = pd.Categorical(ct)
# Categoricals are preferred for efficiency, like factor in R?
adata.obs

# %%
bdata = adata[adata.obs.cell_type == "B"]
bdata

# %%
adata.obsm["X_umap"] = np.random.normal(0, 1, size=(adata.n_obs, 2))
adata.varm["gene_stuff"] = np.random.normal(0, 1, size=(adata.n_vars, 5))
adata.obsm

# %%
# store anything in uns
adata.uns["random"] = [1, 2, 3]
adata.uns

# %%
# add a layer by manual log1p
adata.layers["log_transformed"] = np.log1p(adata.X)
adata

# %%
# nnz return number of non-0 elements in a matrix
# adata.X is not the same as new layer
(adata.X != adata.layers["log_transformed"]).nnz == 0

# %%
# covert to pandas dataframe
adata.to_df(layer="log_transformed")

# %% [markdown]
# ## H5ad I/O

# %%
# anndata is stored on disk as h5ad
# If string columns with small number of categories are not yet categoricals,
# AnnData will auto-transform them to categoricals.
adata.write("my_results.h5ad", compression="gzip")

# %%
# call a shell command, h5ls
#!h5ls 'my_results.h5ad'

# %%
# read a h5ad
adata_new = ad.read_h5ad("my_results.h5ad")
adata_new

# %%
# add some meta data to obs (cells)
obs_meta = pd.DataFrame(
    {
        "time_yr": np.random.choice([0, 2, 4, 8], adata.n_obs),
        "subject_id": np.random.choice(
            ["subject 1", "subject 2", "subject 4", "subject 8"], adata.n_obs
        ),
        "instrument_type": np.random.choice(["type a", "type b"], adata.n_obs),
        "site": np.random.choice(["site x", "site y"], adata.n_obs),
    },
    index=adata.obs.index,  # these are the same IDs of observations as above! index just like rownames of R's dataframe
)

# %%
obs_meta

# %%
# construct a new adata with new obs meta
adata = ad.AnnData(adata.X, obs=obs_meta, var=adata.var)
adata

# %% [markdown]
# ## View and copy
# **Note** Similar to numpy arrays, AnnData objects can either hold actual data or reference another `AnnData` object. In the later case, they are referred to as “view”. Subsetting AnnData objects always returns views, which has two advantages:
# 
# * no new memory is allocated
# 
# * it is possible to modify the underlying AnnData object.
# 
# You can get an actual AnnData object from a view by calling `.copy()` on the view. Usually, this is not necessary, as any modification of elements of a view (calling `.[]` on an attribute of the view) internally calls `.copy()` and makes the view an AnnData object that holds actual data. See the example below.

# %%
adata_view = adata[:5, ["Gene_1", "Gene_3"]]
adata_view

# %%
# copy a new adata
adata_subset = adata[:5, ["Gene_1", "Gene_3"]].copy()
adata_subset

# %%
# a view will be a copy once it tried changing
adata_view.obs["foo"] = range(5)
adata_view.obs

# %% [markdown]
# If a single h5ad file is very large, you can partially read it into memory by using `backed` mode or with the currently experimental `read_elem` API.

# %%
adata = ad.read("my_results.h5ad", backed="r")
adata.isbacked

# %%
# in read-only mode, we cannot damage anything. To proceed with this tutorial, we still need to explicitly close it.
adata.file.close()

# %% [markdown]
# # Scanpy
# `scanpy` for most general scRNA-seq analysis.
# * `squidpy` for spatial transcriptomics.
# * `scirpy` for TCR/BCR analysis.
# * `scflow` for workflows.
# 
# `scvelo` is made like `scanpy`.
# * `pp` module for preprocessing,
# * `tl` module for tools, 
# * `pl` module for plots.

# %%
import scanpy as sc
sc.settings.set_figure_params(dpi=80, facecolor="white")
sc.logging.print_header()

# %%
# get example data
adata = sc.datasets.pbmc3k()
adata

# %%
# visualize most dominant genes in data
sc.pl.highest_expr_genes(adata, n_top=20)

# %%
# classic matrix level QC
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# %%
# pca tool is available both in tl and pp
sc.tl.pca(adata, svd_solver="arpack")
# visualize CST3 expression on PCA plot (featurePlot in seurat)
sc.pl.pca(adata, color='CST3')

# %%
# advanced reduc need compute neighbors
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["CST3", "NKG7", "PPBP"])

# %%



