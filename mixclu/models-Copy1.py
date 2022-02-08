from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from sklearn_extra.cluster import KMedoids
from kmodes.util.dissim import ng_dissim
from kmodes.kmodes import KModes

from prince import FAMD
from .preprocessing import *
from .feats import *
from .base_models import *




def kmoid_grower(df, 
                   cat_columns, 
                   no_of_clusters, 
                   metric       = 'precomputed', 
                   method       = 'pam', 
                   init         = 'build', 
                   max_iter     = 300, 
                   random_state = 4, 
                   df_output    = False):
    
    gower_distance_matrix = calculate_gower_distance(df, cat_columns)
    model_e               = kmedoid_model(gower_distance_matrix, 
                                          no_of_clusters, 
                                          metric       = metric , 
                                          method       = method, 
                                          init         = init, 
                                          max_iter     = max_iter, 
                                          random_state = random_state, 
                                          df_output    = df_output)
    
    return model_e, gower_distance_matrix



def kmoid_grower_umap(df, 
                       cat_columns, 
                       no_of_clusters, 
                       metric       = 'precomputed', 
                       method       = 'pam', 
                       init         = 'build', 
                       max_iter     = 300,
                       n_components = None,     
                       random_state = 4, 
                       df_output    = False):
    
    
    
    if not isinstance(n_components, int):
        n_components      = int(round(np.log(df.shape[1])))
    
    gower_distance_matrix = calculate_gower_distance(df, cat_columns)
    
    # options : diff distance metric, diff umap or pca and then diffe models kmeans and other models
    
    umap_embeddings       = umap.UMAP(random_state = random_state, 
                                      n_components = gower_distance_matrix.shape[0],
                                     ).fit_transform(gower_distance_matrix)

    model_e               = kmedoid_model(umap_embeddings, 
                                          no_of_clusters, 
                                          metric       = metric , 
                                          method       = method, 
                                          init         = init, 
                                          max_iter     = max_iter, 
                                          random_state = random_state, 
                                          df_output    = df_output)
    return model_e



def famd_kemans(df, clusters, 
                random_state = 32, 
                max_iter = 300, 
                verbose = 0, 
                df_output = False):
    
    fmd_matrix = FAMD_2(df)
    
    k_result   = kmeans_model(fmd_matrix, 
                 clusters, 
                 random_state = random_state, 
                 max_iter     = max_iter, 
                 verbose      = verbose, 
                 df_output    = df_output)
    
    return k_result, fmd_matrix