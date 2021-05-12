import random
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import signal
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)

def load_top_expressed_preprocessed_data():
    s745_adata = sc.read("data/sub_0745_miRNA.ec.tab").T
    s743_adata = sc.read("data/s743.tab").T
    s800_adata = sc.read("data/s800.tab").T
    s814_adata = sc.read("data/sub_0814_miRNA.ec.tab").T

    # Note: the mRNA .tab files were earlier preprocessed to drop description col
    # consider moving that here?
    # drop_description("data/sub_0743.genes.no_mt.tpm.rescale.tab", "data/s743.tab")

    # We pull times out of the sample names

    s800_adata.obs['time'] = get_s800_time(s800_adata.obs.index)
    s814_adata.obs['time'] = get_s814_time(s814_adata.obs.index)
    s743_adata.obs['time'] = get_s745_time(s743_adata.obs.index)
    s745_adata.obs['time'] = get_s745_time(s745_adata.obs.index)

    # Split up the 745 runs

    s745_adata_first_run = s745_adata[::2]
    s745_adata_second_run = s745_adata[1::2]
    s814_adata, s814_top = preprocessing(s814_adata, n_expressed=50)
    s800_adata, s800_top = preprocessing(s800_adata)


    s745_1_adata, s745_1_top = preprocessing(s745_adata_first_run, n_expressed=50)
    s745_2_adata, s745_2_top = preprocessing(s745_adata_second_run, n_expressed=50)
    s743_adata, s743_top = preprocessing(s743_adata)
    return (s745_1_top, s745_2_top, s743_top, s800_top, s814_top)


def get_normed_and_zero_filtered_data(filename):
    # TODO perhaps include other normalization options
    df = pd.read_csv(filename, sep = '\t', index_col = 0)
    df = df / df.mean()
    df = df.loc[df.sum(axis=1) != 0,:]
    df = (df
          .assign(sum=df.sum(axis=1))
          .sort_values(by='sum', ascending=False)
          .drop('sum', axis = 1)
         )
    return df

def get_s745_time(names):
    return [float(el.split('_')[1]) for el in names]

def get_s814_time(names):
    return [ 6.25 * float(el.split('_')[0][-2:]) for el in names]

def get_s800_time(names):
    return [ 6.25 * float(el.split('_')[1][-2:]) for el in names]


def plot_one_row_line(df, row_num):
    df.iloc[row_num,:].plot.line()
    
def plot_one_row_scatter(df, row_num, time):
    plt.scatter(df.iloc[row_num,:], time)
    
def dist(x, y):
    # TODO normed difference
    x = x/np.linalg.norm(x)
    y = y/np.linalg.norm(y)
    
    return (x-y).T@(x-y)


"""
    this implementation of k-means takes as input (i) a matrix pd
    with the data points as rows (ii) an integer K representing the number 
    of clusters, and returns (i) a matrix with the K rows representing 
    the cluster centers 
    """
def k_means_from_pd_data(pd, K, maxIters = 300):
   
    centroids = pd[np.random.choice(pd.shape[0], K, replace=False)]
    old_centroids = centroids.copy()
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([dist(x_i, y_k) for y_k in centroids]) for x_i in pd])
        # Update centroids step
        for k in range(K):
            if (C == k).any():                
                centroids[k] = pd[C == k].mean(axis = 0) 
            else: # if there are no data points assigned to this certain centroid
                centroids[k] = pd[np.random.choice(len(pd))] 
        if np.array_equal(centroids, old_centroids):
            print(f'converged on iter: {i}')
            break
        old_centroids = centroids.copy()
    return C, centroids

def add_top_n_expressed_label_to_var(data,top_n):
    cutoff = sorted(data.var.n_counts, reverse=True)[top_n]

    for i in range(0, data.var.shape[0]):
        if data.var.iloc[i,:].n_counts > cutoff:
            data.var.loc[data.var.index[i], 'top_n_expressed'] = True
        else:
            data.var.loc[data.var.index[i], 'top_n_expressed'] = False
    
def drop_description(old_file, new_file):
    df = pd.read_csv(old_file, sep='\t', index_col=0)
    df = df.drop(['description'], axis=1)
    df.to_csv(new_file, sep='\t')

    
    
def preprocessing (adata, n_expressed=200):
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=1e6, exclude_highly_expressed=False)
    sc.pp.log1p(adata)
    


    add_top_n_expressed_label_to_var(adata,n_expressed)

    return adata, adata[:,adata.var.top_n_expressed.isin([True])]

def get_periodgram(sc_data):
    sc_data_zscore = sc.pp.scale(sc_data, max_value=3,copy =True)
    # returns an iterable for each gene, (freq, power, ?I forget?)
    # we only care about power, as the freq vector is the same for all. 
    power = [signal.periodogram(sc_data_z.X[:,i])[1] for i in range(sc_data_z.shape[1])]
    return np.array(power)[0]


def get_mutual_periodgram_genes(adata_miRNA, target_miRNA, adata_mRNA, n_closest=50):
   
    pd_array_mi = get_periodgram(adata_miRNA)
    pd_array_m = get_periodgram(adata_mRNA)
    
    position = list(adata_miRNA.var.index).index(target_miRNA)

    adata_mRNA.var['dist'] = np.nan
    x = pd_array_mi[position]
    for i in range(len(pd_array_m)):
        y = pd_array_m[i]
        adata_mRNA.var.loc[adata_mRNA.var.index[i], 'dist'] = dist(x[3:33],y[3:33])
    
    cutoff = sorted(adata_mRNA.var.dist, reverse=False)[n_closest]
    clost_dist = adata_mRNA[:, adata_mRNA.var.dist < cutoff]
    return clost_dist.var