from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from sklearn_extra.cluster import KMedoids
from kmodes.util.dissim import ng_dissim
from kmodes.kmodes import KModes

from prince import FAMD
from .preprocessing import *
from .feats import *


def kmeans_model(df, 
                 no_of_clusters, 
                 random_state = 42, 
                 max_iter     = 300, 
                 verbose      = 0, 
                 df_output    = False):
    
    """model : 1 : """
    
    """ Pass """
    
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
               df_output    = False,
               scale = None):
    
    """model : 2 
    scale    : minmax, standard or power """
    """ Pass """
    
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
                                       max_iter     = max_iter,
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



def kmeans_famd(df, 
                cat_columns, 
                total_clusters,
                famd_feat_no = 5,
                famd_iter    = 3,
                random_state = 42,
                kmax_iter    = 300, 
                verbose      = 0, 
                df_output    = False):
    
    """model : 4"""
    
    df_famd          = famd_feats(df, cat_columns, famd_feat_no, famd_iter)
    
    result           = kmeans_model(df              = df_famd, 
                                     no_of_clusters = total_clusters, 
                                     random_state   = random_state, 
                                     max_iter       = kmax_iter, 
                                     verbose        = verbose, 
                                     df_output      = df_output)
    return result




def kmedoid_model(embeddings, 
                  num_clusters, 
                  metric       = 'precomputed', 
                  method       = 'pam', 
                  init         = 'build', 
                  max_iter     = 300, 
                  random_state = 32, 
                  df_output    = True):
    
    
    model        = KMedoids(n_clusters = num_clusters, 
                            metric     = metric,
                            method     = method, 
                            init       = init, 
                           max_iter    = max_iter, 
                           random_state=random_state).fit(embeddings)
    
    labels       = gower_clusterer.labels_
    if df_output:
        scores                 = pd.DataFrame()
        scores['clusters']     = labels
        
        return scores, model
    
    return {'clusters': labels, 
            'model'   : model}





def kmodels_model(df, init='Huang', 
                  n_init=1, 
                  verbose=2, 
                  cat_dissim = None, 
                  df_output = True):
    
    if cat_dissim:
        model = KModes(n_clusters=K,
                       init='Huang',
                       n_init=1,
                       cat_dissim=ng_dissim,
                       verbose=2)
    else:
        model = KModes(n_clusters=K,
                       init='Huang',
                       n_init=1,
                       verbose=2)
        
    clusters = model.fit_predict(df)
    
    if df_output:
        df['clusters']   = list(clusters)
        return df, model
    
    return {'clusters': clusters, 
            'model'   : model}
