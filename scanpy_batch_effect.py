# Based on scanpy integration tutorial
# %%
import scanpy as sc
import pandas as pd
import seaborn as sns

# verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.verbosity = 1
sc.logging.print_versions()

# %%
# ingest need a ref dataset to integrate sample data
# this is an earlier version of the dataset from the pbmc3k tutorial
adata_ref = sc.datasets.pbmc3k_processed()
adata = sc.datasets.pbmc68k_reduced()

# %%
# To use sc.tl.ingest, the datasets need to be defined on the same vars
var_names = adata_ref.var_names.intersection(adata.var_names)
adata_ref = adata_ref[:, var_names]
adata = adata[:, var_names]

# %%
# we need model trained on the ref data
sc.pp.pca(adata_ref)
sc.pp.neighbors(adata_ref)
sc.tl.umap(adata_ref)

sc.pl.umap(adata_ref, color='louvain')

# %%
# map labels and embeddings from adata_ref to adata based on a chosen representation
# Here, we use adata_ref.obsm['X_pca'] to map cluster labels and the UMAP coordinates
sc.tl.ingest(adata, adata_ref, obs='louvain')
adata.uns['louvain_colors'] = adata_ref.uns['louvain_colors']  # fix colors
sc.pl.umap(adata, color=['louvain', 'bulk_labels'], wspace=0.5)

# %%
# concat ref and sample data
adata_concat = sc.concat([adata,adata_ref],label='batch_categories',keys=['new','ref'])
# transform lonvain to category type
adata_concat.obs.louvain = adata_concat.obs.louvain.astype('category')
# fix category ordering
adata_concat.obs.louvain.cat.reorder_categories(adata_ref.obs.louvain.cat.categories)
# fix category colors
adata_concat.uns['louvain_colors'] = adata_ref.uns['louvain_colors']

# %%
# ingested data only modified pca & umap
sc.pl.umap(adata_concat, color=['batch_categories', 'louvain'])

# %%
# bbknn
sc.tl.pca(adata_concat)
# simply supply batch_key to bbknn
sc.external.pp.bbknn(adata_concat, batch_key='batch')  # running bbknn 1.3.6
sc.tl.umap(adata_concat)
sc.pl.umap(adata_concat, color=['batch_categories', 'louvain'])

# %%
# ingest can be used by choose one batch as ref, reduc and map to all other batches