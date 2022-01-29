from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes

from prince import FAMD
from .preprocessing import *
from .feats import *


def kmeans_model(df, 
                 no_of_clusters, 
                 random_state = 42, 
                 max_iter     = 300, 
                 verbose      = 0, 
                 df_output    = False):
    
    """model : 1"""
    
    model     = KMeans(n_clusters   = no_of_clusters, 
                   random_state     = random_state,
                   max_iter         = max_iter, 
                   verbose          = 0).fit(df)
    
    
    clusters  = model.labels_
    
    if df_output:
        df['clusters']   = list(clusters)
        return df, model
    
    return {'clusters': clusters, 
            'model'   : model}


def kmeans_onehot_mix(df, 
               cat_columns, 
               total_clusters,
               random_state = 42,
               max_iter     = 300, 
               verbose      = 0, 
               df_output    = False
               scale = None):
    
    """model : 2 
    scale    : minmax or standard """
    
    df_one, cat_col      = one_hot_df(df, cat_columns, True)
    if scale:
        print(f'scaler : {scale}')
        df_one           = normalize_df(df_one, scale)
        
    result               = kmeans_model(df              = df_one, 
                                         no_of_clusters = total_clusters, 
                                         random_state   = random_state, 
                                         max_iter       = max_iter, 
                                         verbose        = verbose, 
                                         df_output      = df_output)
    return result



def k_prot_model(df, 
           cat_cols, 
           total_clusters, 
           init_method  = 'Huang',
           random_state = 42,
            max_iter    = 300, 
            verbose     = 0, 
            df_output   = False, 
            n_init      = 10):
    

    
    
    """model : 3 
    Huang or Cao"""
    
    df_train, cat_num    = k_proto_data(df, cat_cols)
    
    kproto               = KPrototypes(n_clusters   = total_clusters, 
                                       max_iter     = max_iter
                                       init         = init_method, 
                                       random_state = random_state,
                                        verbose     = verbose,
                                        n_jobs      =-1,
                                        n_init      = n_init)
    
    
    clusters             = kproto.fit_predict(df_train, 
                                              categorical=cat_num)
    
    if df_output:
        df_train['clusters']   = list(clusters)
        return df_train, kproto
    
    return {'clusters': clusters, 
            'model'   : kproto}