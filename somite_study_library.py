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
    power = [signal.periodogram(sc_data_zscore.X[:,i])[1] for i in range(sc_data_zscore.shape[1])]
    return np.array(power)


def get_mutual_periodgram_genes(adata_miRNA, target_miRNA, adata_mRNA, n_closest=50):
   
    pd_array_mi = get_periodgram(adata_miRNA)
    pd_array_m = get_periodgram(adata_mRNA)
    
    position = list(adata_miRNA.var.index).index(target_miRNA)

    adata_mRNA.var['dist'] = np.nan
    x = pd_array_mi[position]
    for i in range(len(pd_array_m)):
        y = pd_array_m[i]
        #print(x,y,adata_mRNA.shape,i)
        adata_mRNA.var.loc[adata_mRNA.var.index[i], 'dist'] = dist(x[3:33],y[3:33])
    
    cutoff = sorted(adata_mRNA.var.dist, reverse=False)[n_closest]
    clost_dist = adata_mRNA[:, adata_mRNA.var.dist < cutoff]
    return clost_dist.var

def plot_combine_targetMiR_ComMR(common_genes_set,miR_set,mR_set):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    time = mR_set.obs.time
    
    for g in common_genes_set:
        position = list(mR_set.var.index).index(g)
        y = mR_set.X[:,position]
        ax.plot(time,y/np.linalg.norm(y), label=mR_set.var.index[position].split(',')[0])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
    
    position1 = list(miR_set.var.index).index('hsa-miR-10a-5p')
    y = miR_set.X[:,position1]
    ax.plot(time,y/np.linalg.norm(y), label=miR_set.var.index[position1].split(',')[0])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def plot_targetmiRNA_group(miRNA,C):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    time = miRNA.obs.time

    for i in range(len(C)):
        if miRNA.var.index[i] =='hsa-miR-10a-5p':
            a = C[i]
    for i in range(len(C)):
        if C[i]==a:
            y = miRNA.X[:,i]
            ax.plot(time,y/np.linalg.norm(y), label=miRNA.var.index[i].split(',')[0])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
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

def get_normed_and_zero_filtered_data_jws(filename, alg = 'cpm', quantile=None):
    df = pd.read_csv(filename, sep = '\t', index_col = 0)
    df = df.loc[df.sum(axis=1) != 0,:]
    assert alg in ['cpm', 'qbr']
    if alg == 'cpm':
         df = df / df.mean() * 1e6
    if alg == 'qbr':
        quantile = int(quantile)
        assert quantile > 0 and quantile < 100
      
    df = (df
          .assign(sum=df.sum(axis=1))
          .sort_values(by='sum', ascending=False)
          .drop('sum', axis = 1)
         )
    return df

def get_series_for_var(var_of_interest, adata):
    position = list(adata.var.index).index(var_of_interest)
    return adata.X[:,position]

def draw_single_miRNA(miRNA_of_interest, data_sets):
    fig, ax = plt.subplots(figsize=(15,10))

    for label, miRNA_set in data_sets.items():
        y = get_series_for_var(miRNA_of_interest,miRNA_set)
        x = miRNA_set.obs.time
        ax.plot(x,y/np.linalg.norm(y), label=label, linewidth=2.5,marker='o')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Normalized expression')
    plt.title(f"{miRNA_of_interest} Under Three Time Courses")
    return (fig, ax)

def prep_data(adata,K_start, K_end):
    return adata[K_start:len(adata.obs)-K_end-1]
   

def plot_2_periodogram(miRNA_set, series_1, series_2,time):
    
    position1 = list(miRNA_set.var.index).index(series_1)
    position2 = list(miRNA_set.var.index).index(series_2)
    
    f, Pxx_den = signal.periodogram(miRNA_set[:,position1].X.T, fs = 1/time/60)
    plt.scatter(f, Pxx_den.T,label=series_1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([1e-2, 1e3])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    f, Pxx_den = signal.periodogram(miRNA_set[:,position2].X.T, fs = 1/time/60)
    plt.scatter(f, Pxx_den.T,label=series_2)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([1e-2, 1e3])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    
    return
    

def get_similarity(miRNA_set, series_1, series_2,time):
    # concerns:
    #  1) similarity should be bounded to [0,1] (possbly [-1,1]) but certainly no big negative numbers!
    #  normalization should take care of this. 
    #  2) filter - complicted because we need to choose freq fiter, and then implement given out time 
    #  parameter to this function. 
    
    
    position1 = list(miRNA_set.var.index).index(series_1)
    position2 = list(miRNA_set.var.index).index(series_2)
    
    
    pd_array = get_periodgram(miRNA_set)
    series_1_array = pd_array[position1]
    series_2_array = pd_array[position2]
    for i in (0,len(series_1_array)-1): # what if we just itereate over frequencies of interest? (0.00013, 0.0033)
        dist = np.sum ((series_1_array[i] - series_2_array[i])**2)
        

    # 2) filter to frequencies of interest (not too fast, <= 30min)
    # 3) How close are they? 
    # 3a Naive : sum ((series_1_pd_i :- series_2_pd_i)**2)
    # 3b) maybe normalize first? (normalize in frequency space)
    # 3c) Apply filter to only use some frequencies?
    
    return 1 - dist**.5
   
if __name__ == '__main__':
    ''' 
    This is for debugging. Have to remember to be careful when pasting back 
    and forth to the jupyter notebook, because the namespace there always start
    with lib.<thingy>
    '''
    sub_0745_I, sub_0745_II, sub_0743, sub_0800, sub_0814 = load_top_expressed_preprocessed_data()
    series_1 = get_series_for_var('hsa-miR-125a-5p', sub_0814)
    series_2 = get_series_for_var('hsa-miR-10a-5p', sub_0814)
    get_similarity(series_1, series_2)
