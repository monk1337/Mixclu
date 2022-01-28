from pandas as pd
from numpy  as np

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from random import sample
from numpy.random import uniform
from math import isnan


def cobj(df, cat_columns):
    
    """convert columns to object type """
    for col_name in cat_columns:
        df[col_name] = df[col_name].astype(object)
    return df


def one_hot_df(df, cat_columns):
    
    other_col          = [col for col in df.columns 
                                if col not in cat_columns]
    df                 = pd.get_dummies(df, columns=cat_columns)
    updated_cat_columns= list(set(df.columns) - set(other_col))
    df                 = cobj(df, updated_cat_columns)
    return df


def famd_feats(df, cat_columns,
                   no_of_components, 
                   n_iter = 3):
    
    df     = cobj_b(df, cat_columns)
    famd   = FAMD(n_components = no_of_components, 
                  n_iter       = n_iter).fit(df)
    result = famd.row_coordinates(df)
    return result


def hopkins(X):
    
    """ Need to try this too 
    https://pyclustertend.readthedocs.io/en/latest/ """
    
    d = X.shape[1]
    n = len(X) # rows
    m = int(0.1 * n) # heuristic from article [1]
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H



def pca_feats(df, dim = 3):
    
    Num_features        = df.select_dtypes(include=[np.number]).columns
    x                   = df[Num_features]
    
    
    if dim <= min(x.shape[0], x.shape[1]):
        dim = dim
    else:
        dim = min(x.shape[0], x.shape[1])
        
    print('total pca dim ', dim)
    
    pca                 = PCA(n_components = dim, whiten=True)
    principalComponents = pca.fit_transform(x)

    cum_explained_var   = []
    for i in range(0, len(pca.explained_variance_ratio_)):
        if i == 0:
            cum_explained_var.append(pca.explained_variance_ratio_[i])
        else:
            cum_explained_var.append(pca.explained_variance_ratio_[i] + 
                                     cum_explained_var[i-1])
    
    columns        = [f'pca_{col_name}' for col_name in range(dim)]
    cev            = {columns[out] : [cum_explained_var[out]] 
                      for out in range(len(cum_explained_var))}
    cev            = pd.DataFrame(cev)
    pca_df         = pd.DataFrame(data = principalComponents,
                                  columns = columns)
    
    final_df_cols  = list(set(df.columns) - set(Num_features))
    final_df_cols  = df[final_df_cols]
    final_df       = pd.concat([final_df_cols, pca_df], axis = 1)
    
    return final_df, cev