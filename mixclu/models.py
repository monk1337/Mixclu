from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from sklearn_extra.cluster import KMedoids
from kmodes.util.dissim import ng_dissim
from kmodes.kmodes import KModes

from prince import FAMD
from .preprocessing import *
from .feats import *
from .base_models import *
from .embeddings.umap_embd import *




def kmoid_grower(df, 
                   cat_columns, 
                   no_of_clusters,
                   max_iter     = 300, 
                   random_state = 4, 
                   df_output    = False):
    
    gower_distance_matrix = calculate_gower_distance(df, cat_columns)
    model_e               = kmedoid_model(gower_distance_matrix, 
                                          no_of_clusters, 
                                          metric       = 'precomputed', 
                                          method       = 'pam', 
                                          init         = 'build', 
                                          max_iter     = max_iter, 
                                          random_state = random_state, 
                                          df_output    = df_output)
    
    return model_e, gower_distance_matrix



def kmoid_grower_umap(df, 
                       cat_columns, 
                       no_of_clusters,
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
                                          metric       = 'precomputed' , 
                                          method       = 'pam', 
                                          init         = 'build', 
                                          max_iter     = max_iter, 
                                          random_state = random_state, 
                                          df_output    = df_output)
    
    return umap_embeddings, gower_distance_matrix, model_e



def famd_kemans(df, 
                cat_columns, 
                clusters, 
                random_state = 32, 
                max_iter = 300, 
                verbose = 0, 
                df_output = False):
    
    df         = cobj(df, cat_columns)
    fmd_matrix = FAMD_2(df)
    
    k_result   = kmeans_model(fmd_matrix, 
                 clusters, 
                 random_state = random_state, 
                 max_iter     = max_iter, 
                 verbose      = verbose, 
                 df_output    = df_output)
    
    return k_result, fmd_matrix



def umap_embd_model(df, 
                    cat_columns, 
                    no_of_clusters, 
                    random_state = 32, 
                    max_iter     = 300, 
                    verbose      = 0, 
                    df_output    = False):
    
    
    df             = cobj(df, cat_columns)
    _, emd         = umap_reduce(df)
    
    UMAP_clusterer = kmeans_model(emd, 
                                 no_of_clusters = no_of_clusters, 
                                 random_state = random_state, 
                                 max_iter     = max_iter, 
                                 verbose      = verbose, 
                                 df_output    = df_output)
    
    return UMAP_clusterer, emd





def Mirkin_model(df, 
                 cat_columns, 
                 no_of_clusters, 
                 random_state = 32, 
                 max_iter = 300, 
                 verbose =0, 
                 df_output = False):
    
    
    
    z_data        = zscore_preprocessing(df, 
                                         cat_columns, 
                                         cat_col_zscore = True)
    
    mirkin_result = kmeans_model(z_data, 
                                 no_of_clusters = no_of_clusters, 
                                 random_state = random_state, 
                                 max_iter     = max_iter, 
                                 verbose      = verbose, 
                                 df_output    = df_output)
    
    
    
    return z_data, mirkin_result



def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]


def zscore_preprocessing(df, cat_columns, 
                         cat_col_zscore = True):

    df             = cobj(df, cat_columns)
    continous_data = df.copy()
    numeric_cols   = continous_data.select_dtypes(include=np.number)
    cat_cols       = continous_data.select_dtypes(include='object')
    
    if cat_col_zscore:

        # numeric process
        normalized_df = calculate_zscore(continous_data, numeric_cols)
        normalized_df = normalized_df[numeric_cols.columns]

        # categorical process
        cat_one_hot_df, one_hot_cols = one_hot_encode(continous_data, cat_cols)
        cat_one_hot_norm_df          = calculate_zscore(cat_one_hot_df, one_hot_cols)

        # Merge DataFrames
        processed_df = pd.concat([normalized_df, cat_one_hot_norm_df], axis=1)
    else:
        
        norm_num_cols = calculate_zscore(numeric_cols, numeric_cols)
        processed_df  = pd.concat([norm_num_cols, cat_cols], axis=1)

    return processed_df




def k_prototype_z_model(df, 
                        cat_columns, 
                        num_of_clusters, 
                        init_method = 'Cao', 
                        random_state= 32, 
                        max_iter    = 300, 
                        n_jobs      = 1, 
                        verbose     = 0, 
                        df_output   = False, 
                        n_init      = 10):
    
    
    z_data              = zscore_preprocessing(df, 
                                               cat_columns, 
                                               cat_col_zscore = False)
    
    categorical_indices = column_index(z_data, cat_columns)
    
    model_result = k_prot_model(z_data, 
                               list(cat_columns), 
                               num_of_clusters, 
                               init_method   = init_method,
                               random_state  = random_state,
                                max_iter     = max_iter,
                                n_jobs       = n_jobs,
                                verbose      = verbose, 
                                df_output    = df_output, 
                                n_init       = n_init)
    
    return z_data, model_result



def kmodes_cont(df, 
                bin_con_columns,
                n_of_clusters,
                bin_bins         = 5, 
                bin_drop_cols    = True, 
                bin_encode       = False, 
                mode_init_method = 'Cao', 
                mode_n_init      = 10, 
                mode_verbose     = 0, 
                mode_random_state= 32, 
                mode_n_jobs      = 1, 
                mode_cat_dissim  = None, 
                mode_df_output   = None):

    recode_df = get_knn_bins(df, bin_con_columns, 
                     bins= bin_bins, 
                     drop_cols=bin_drop_cols, 
                     encode = bin_encode)


    k_mode_result = kmodes_model(recode_df, 
                       n_of_clusters,
                      init         = mode_init_method,
                      n_init       = mode_n_init, 
                      verbose      = mode_verbose,
                      random_state = mode_random_state,
                      n_jobs       = mode_n_jobs,
                      cat_dissim = mode_cat_dissim, 
                      df_output  = mode_df_output)
    
    return k_mode_result



def categorical_embedding_model(df, 
                                bin_con_columns, 
                                no_of_clusters = 4,
                                bin_bins = 5,
                                bin_drop_cols = False,
                                bin_encode = False,
                                n_neighbors=10, 
                                min_dist=0.1, 
                                random_state = 32, 
                                max_iter = 300, 
                                verbose = 0, 
                                df_output = False):
    
    
    recode_df = get_knn_bins(df, bin_con_columns, 
                 bins= bin_bins, 
                 drop_cols=bin_drop_cols, 
                 encode = bin_encode)
        
    
    cat_cols      = recode_df.select_dtypes(include='object')
    df_one_hot, _ = one_hot_encode(recode_df, cat_cols)

    umap_embedding = (umap
                    .UMAP(metric='sokalsneath', 
                          n_neighbors=n_neighbors,
                          min_dist=min_dist)
                    .fit_transform(df_one_hot))

        
    result        = kmeans_model(umap_embedding, 
                                 no_of_clusters = no_of_clusters, 
                                 random_state = random_state, 
                                 max_iter     = max_iter, 
                                 verbose      = verbose, 
                                 df_output    = df_output)
    
    return umap_embedding, result




def graph_clustering(df, con_columns,
                     bins = 5, 
                     bin_drop_columns = False, 
                     bin_encode = False, 
                     df_output = False):
    
    recode_df = get_knn_bins(df, 
                             con_columns, 
                             bins= bins, 
                             drop_cols=bin_drop_columns, 
                             encode = bin_encode)
    
    graph                = convert_df_to_sgraph_network(recode_df)
    communities_dict     = community_louvain.best_partition(graph)
    communites, features = convert_community_output_to_df(communities_dict)
    graph_comunities     = merge_clusters_back_df(recode_df, communites)
    
    if df_output:
        df['clusters'] = list(graph_comunities["Clusters"].values)
        return df
    else:
        return {'clusters': list(graph_comunities["Clusters"].values)}