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
               scale = None):
    
    """model : 2 
    scale    : minmax or standard """
    
    df_one, cat_col      = one_hot_df(df, cat_columns, True)
    if scale:
        print(f'scaler : {scale}')
        df_one           = normalize_df(df_one, scale)
    result               = kmeans_model(df_one, total_clusters)
    return result['clusters'], result['model']



def k_prot_model(df, 
           cat_cols, 
           total_clusters, 
           init_method = 'Huang'):
    
    
    """model : 3 
    Huang or Cao"""
    
    df_train, cat_num    = k_proto_data(df, cat_cols)
    
    kproto               = KPrototypes(n_clusters=total_clusters, 
                                                 init=init_method, 
                                                 random_state=42, 
                                                 verbose=0,
                                                 n_jobs=-1,
                                                 n_init=50)
    
    clusters             = kproto.fit_predict(df_train, 
                                              categorical=cat_num)
    return clusters, kproto