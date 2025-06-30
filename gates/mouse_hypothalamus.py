import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
from sklearn.metrics.cluster import adjusted_rand_score
import sys
from statsmodels.stats.multitest import multipletests
from matplotlib.pyplot import rc_context
sys.path.append(r'D:\Pycharm\Py_Projects\gates')
from Train_GATES import train_GATES
from utils import Cal_Spatial_Net, Stats_Spatial_Net, mclust_R, Cal_Gene_Similarity_Net
os.environ['R_HOME'] = 'D:/anaconda3/R-4.0.3'
os.environ['R_USER'] = 'D:/anaconda3/envs/GraphST/Lib/site-packages/rpy2'
seed = 2020
section_id = 'MERFISH_mouse_hypothalamic'
data_root = 'E:/Spatial_data/data'
adata = sc.read_h5ad("E:\Spatial_data\data\MERFISH_mouse_hypothalamic\merfish-0.14.h5ad")
adata.var_names_make_unique()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
Cal_Spatial_Net(adata, model="Radius", rad_cutoff=150)  # model="KNN", k_cutoff=6
Cal_Gene_Similarity_Net(adata, k_neighbors=7, metric='cosine', verbose=True)  # cosine, euclidean, manhattan
Stats_Spatial_Net(adata,plt_show=False)
adata = train_GATES(adata, mod='spatial-similarity', alpha=0.01, n_epochs=500, verbose=False, random_seed=seed)
sc.pp.neighbors(adata, use_rep='GATES')#可能是 train_GATES 函数的产物，可能是一种低维表示、特征表示或网络表示。
sc.tl.umap(adata)
adata = mclust_R(adata, used_obsm='GATES', num_cluster=8)
obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(adata.obs['region'], adata.obs['mclust'])
sc.pl.embedding(adata, basis="spatial", color='mclust', s=20, show=False)#,


