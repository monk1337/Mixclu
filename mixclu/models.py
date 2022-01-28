from sklearn.cluster import KMeans
from prince import FAMD



def kmeans_model(data, 
                 no_of_clusters, 
                 random_state = 42, 
                 max_iter     = 300, 
                 verbose      = 0):
    
    """model : 1"""
    
    
    model     = KMeans(n_clusters   = no_of_cluster, 
                   random_state     = random_state,
                   max_iter         = max_iter, 
                   verbose          = 0).fit(data)
    
    
    clusters  = model.labels_
    return {'clusters': clusters, 
            'model'   : model}

            