from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score

import io
import math
import pandas as pd
import numpy as np
import math



def evaluate_clusters(scores,  preds, labels, name='', X=None):
    
    if X is not None:

        silhouette = silhouette_score(X, preds, metric='euclidean')
        cal_har = calinski_harabasz_score(X, preds)
        dav_bould = davies_bouldin_score(X, preds)

        adj_mut_info = adjusted_mutual_info_score(labels, preds, average_method='arithmetic')
        adj_rand = adjusted_rand_score(labels, preds)

        content = {'Algorithm':name,
                   'Silhouette':silhouette,
                   'Calinski_Harabasz':cal_har,
                   'Davis Bouldin':dav_bould,
                   'Adjusted_Mutual_Info':adj_mut_info,
                   'Adjusted_Rand_Score':adj_rand}

        scores = scores.append(content, ignore_index = True)

    else:

        adj_mut_info = adjusted_mutual_info_score(labels, preds, average_method='arithmetic')
        adj_rand = adjusted_rand_score(labels, preds)

        content = {'Algorithm':name,
                   'Silhouette':np.NaN,
                   'Calinski_Harabasz':np.NaN,
                   'Davis Bouldin':np.NaN,
                   'Adjusted_Mutual_Info':adj_mut_info,
                   'Adjusted_Rand_Score':adj_rand}

        scores = scores.append(content, ignore_index = True)

    return scores