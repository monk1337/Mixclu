import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from random import sample
from numpy.random import uniform
from math import isnan
from prince import FAMD


def cobj(df, cat_columns):
    
    """convert columns to object type """
    for col_name in cat_columns:
        df[col_name] = df[col_name].astype(object)
    return df


def one_hot_df(df, cat_columns, cat_names = False):
    
    other_col          = [col for col in df.columns 
                                if col not in cat_columns]
    df                 = pd.get_dummies(df, columns=cat_columns)
    updated_cat_columns= list(set(df.columns) - set(other_col))
    df                 = cobj(df, updated_cat_columns)
    
    if cat_names:
        return df, updated_cat_columns
    return df


# def famd_feats(df, cat_columns,
#                    no_of_components, 
#                    n_iter = 3):
    
#     df     = cobj(df, cat_columns)
#     famd   = FAMD(n_components = no_of_components, 
#                   n_iter       = n_iter).fit(df)
#     result = famd.row_coordinates(df)
#     print('explained_inertia : ', np.sum(famd.explained_inertia_))
#     return result, np.sum(famd.explained_inertia_)

def factor(data, comp = None):
    
    if comp:
        total_components = comp
    else:
        total_components = data.shape[1]
    
    famd   = FAMD(n_components = total_components).fit(data)
    res    = np.sum(famd.explained_inertia_)
    if round(res, 2) >= 0.95 and round(res, 2) != 1.0 or comp == 0:
        return famd
    else:
        return factor(data, total_components-1)



def famd_feats(df, cat_columns,
                   no_of_components = 'auto', 
                   n_iter = 3):
    
    
    df     = cobj(df, cat_columns)
    if no_of_components == 'auto':
        famd   = factor(df)
    else:
        famd   = FAMD(n_components = no_of_components, 
                      n_iter       = n_iter).fit(df)
    
    
    result = famd.row_coordinates(df)
    print('explained_inertia : ', round(np.sum(famd.explained_inertia_), 2))
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



def pca_feats(df, con_columns, dim = 'auto'):
    
    
    #     Num_features        = df.select_dtypes(include=[np.number]).columns
    #     x                   = df[Num_features]
    
    x    = df[con_columns]
    
    if dim  == 'auto':
        dim = 0.95
    else:
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
    
    final_df_cols  = list(set(df.columns) - set(con_columns))
    final_df_cols  = df[final_df_cols]
    final_df       = pd.concat([final_df_cols, pca_df], axis = 1)
    
    return final_df, cev