# Basically based on scanpy pp tutorial
# %%
import numpy as np
import pandas as pd
import scanpy as sc

# %%
skpath = "/home/gjsx/work/results/seekgene_crc/expr_mat/"

def read_scvelo(path,batch):
    spl = sc.read_mtx(path + 'spliced.mtx.gz').X.transpose()
    unspl = sc.read_mtx(path + 'unspliced.mtx.gz').X.transpose()
    obsn = pd.read_csv(path + 'barcodes.tsv.gz',
                      delimiter='\t',
                      names=['id'])
    varn = pd.read_csv(path + 'features.tsv.gz',
                      delimiter='\t',
                      names=['id','name','foo'])
    adata = sc.AnnData(spl)
    adata.obs_names = [f"{batch}_{i}" for i in obsn['id']]
    adata.var_names = varn['name']
    adata.obs['batch'] = batch
    adata.layers['spliced'] = adata.X
    adata.layers['unspliced'] = unspl
    return adata
    
# %%
crc22 = read_scvelo(skpath + 'crc221125-seek/star_solo/Solo.out/Velocyto/filtered/', 'crc221125')

# %%
crc230304 = read_scvelo(skpath + 'crc230304_seek/star_solo/Solo.out/Velocyto/filtered/', 'crc230304') 

# %%
crc230309 = read_scvelo(skpath + 'crc230309_seek/star_solo/Solo.out/Velocyto/filtered/', 'crc230309')

# %% filter out low-expression genes
print(crc22.n_vars)
sc.pp.filter_genes(crc22,min_cells=3)
print(crc22.n_vars)

# %%
print('unfiltered genes: '+str(crc230309.n_vars))
sc.pp.filter_genes(crc230309,min_cells=3)
print('filtered genes: '+str(crc230309.n_vars))

# %%
print('unfiltered genes: '+str(crc230304.n_vars))
sc.pp.filter_genes(crc230304,min_cells=3)
print('filtered genes: '+str(crc230304.n_vars))

# %% check if vars are duplicated
print([crc22.var.index.is_unique, crc230304.var.index.is_unique, crc230309.var.index.is_unique])
crc22.var_names_make_unique()
crc230304.var_names_make_unique()
crc230309.var_names_make_unique()

# %% merge batches -------------
adata = sc.concat([crc22,crc230304,crc230309])
adata

# %% filter out low expression cells
sc.pp.filter_cells(adata,min_genes=200)
adata

# %% check most expressed genes in data
sc.pl.highest_expr_genes(adata,n_top=20)

# %% annotate the group of mitochondrial genes as 'mt'
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# %% visualize qc stat (n_genes_by_counts = n_genes)
sc.pl.violin(adata, ['n_genes', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)

# %% filter out high mito perc cells (8289 -> 7530)
adata = adata[adata.obs.pct_counts_mt < 10, :]
adata

# %% normalize, log1p & identify high var genes
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)

# %% save the current state of adata to its raw attr
# can use raw_adata = adata.raw.to_adata() to retreive raw state
adata.raw = adata

# %% keep only variable genes (actually not necessary, since pca & neighbor finding auto use var genes)
adata = adata[:, adata.var.highly_variable]

# %% regress out effect of total counts & mt perc
# will densify sparse matrix, cost 22s
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

# %% scale to variance unit and clip values exceeding 10*sd
sc.pp.scale(adata, max_value=10)

# %% PCA -------
sc.tl.pca(adata)
sc.pl.pca(adata, color='FCGR2B')

# %% knee plot of pcs
sc.pl.pca_variance_ratio(adata, log=True)

# %% save (default) h5ad
adata.write('data/seekgene/3sample.h5ad')

# %% read h5ad
adata = sc.read('data/seekgene/3sample.h5ad')
adata

# %% compute neighbor graph from pca
sc.pp.neighbors(adata)

# %% embedding neighbor graph by adv reduc
# UMAP is potentially more faithful to the global connectivity of the manifold than tSNE,
# i.e., it better preserves trajectories
sc.tl.umap(adata)
sc.pl.umap(adata, color=['FCGR2B'])

# %% default feature map use raw (log) expr
# set use_raw=False to plot by regressed scaled expr
sc.pl.umap(adata, color=['FCGR2B'], use_raw=False)

# %% Leiden graph-clustering
# !mamba install leidenalg
# default res=1
sc.tl.leiden(adata)
sc.pl.umap(adata, color=['leiden']) # dimplot the same as featureplot

# %%
# find all cluster markers
adata.uns['log1p']['base'] = None
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups_dotplot(adata,n_genes=1)

# %%
# good in pub
sc.pl.rank_genes_groups_stacked_violin(adata,n_genes=2)
# matrixplot is like heatmap frame in dotplot data
#sc.pl.rank_genes_groups_matrixplot(adata)
# tracksplot is like thin barplot per cell
#sc.pl.rank_genes_groups_tracksplot(adata,n_genes=2)
# seen in pub
#sc.pl.rank_genes_groups_heatmap(adata,n_genes=3)

# %%
# or using logreg method to regress out covar
sc.tl.rank_genes_groups(adata, 'leiden', method='logreg')

# %%
# show top 5 genes (stored in .uns) in dataframe
pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)

# %%
# compare cluster 0 to cluster 1
sc.tl.rank_genes_groups(adata, 'leiden', groups=['0'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)

# %%
# simple violin on certain genes
sc.pl.violin(adata, ['CST3', 'NKG7', 'PPBP'], groupby='leiden')

# %%
# manually add cell type names
new_cluster_names = [
    'CD4 T', 'CD14 Monocytes',
    'B', 'CD8 T',
    'NK', 'FCGR3A Monocytes',
    'Dendritic', 'Megakaryocytes']
adata.rename_categories('leiden', new_cluster_names)
sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, save='.pdf')

marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']
sc.pl.dotplot(adata, marker_genes, groupby='leiden')

# %%
# `compression='gzip'` saves disk space, but slows down writing and subsequent reading
adata.write(adata, compression='gzip')

# Export single fields of the annotation of observations
# adata.obs[['n_counts', 'louvain_groups']].to_csv(
#     './write/pbmc3k_corrected_louvain_groups.csv')

# Export single columns of the multidimensional annotation
# adata.obsm.to_df()[['X_pca1', 'X_pca2']].to_csv(
#     './write/pbmc3k_corrected_X_pca.csv')

# Or export everything except the data using `.write_csvs`.
# Set `skip_data=False` if you also want to export the data.
# adata.write_csvs(results_file[:-5], )