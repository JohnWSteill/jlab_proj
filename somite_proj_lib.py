def get_normed_and_zero_filtered_data(filenamei, alg = 'cpm', quantile=None):
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
