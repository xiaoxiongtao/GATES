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
from matplotlib.patches import Patch
from PIL import Image, ImageEnhance
import seaborn as sns
import matplotlib.pyplot as plt
os.environ['R_HOME'] = 'D:/anaconda3/R-4.0.3'
os.environ['R_USER'] = 'D:/anaconda3/envs/GraphST/Lib/site-packages/rpy2'
seed = 2020
section_id_1 = 'Sagittal-Anterior'
data_root = 'E:/Spatial_data/data/Mouse Brain Serial Section 2'
input_dir_1 = os.path.join(data_root, section_id_1)
adata = sc.read_visium(path=input_dir_1, count_file='V1_Mouse_Brain_Sagittal_Anterior_Section_2_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()
Cal_Spatial_Net(adata, model="Radius", rad_cutoff=150)
Cal_Gene_Similarity_Net(adata, k_neighbors=7, metric='cosine', verbose=True)
Stats_Spatial_Net(adata, plt_show=False)
adata = train_GATES(adata, mod='spatial-similarity', alpha=0.01, n_epochs=500, verbose=False, random_seed=seed)
sc.pp.neighbors(adata, use_rep='GATES')
sc.tl.umap(adata)
adata = mclust_R(adata, used_obsm='GATES', num_cluster=10)
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False,groups=['5','6'],dpi=300,show=False)
sc.pl.spatial(adata,img_key="hires",color=['Ttr','Ppp1r1b'],alpha_img=0.5,alpha=1,size=1.3,show=False)
sc.pl.spatial(adata,img_key="hires",color=['mclust'],title=['Ours'],show=False)