import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def get_dummy_data(n_samples, 
                   columns, 
                   cat_col, 
                   missing_values = None, 
                   id_cols        = None):
    
    """generate dummy mix data types df """
    
    X, y          = make_blobs(n_samples=n_samples, centers=3, n_features=columns, random_state=12)
    
    columns_name  = [f'X{col_name}' for col_name in range(columns)]
    X             = pd.DataFrame(X, columns=columns_name)
    cat_columns   = np.random.choice(columns_name, cat_col, replace=False)
    for cat_co in cat_columns:
        X[cat_co]       = np.where(X[cat_co] < 0, 0, 1)

    con_feats = list(set(columns_name) - set(cat_columns))
    cat_feats = cat_columns
    scale = StandardScaler()
    X[con_feats] = scale.fit_transform(X[con_feats])
    
    if missing_values:
        X = X.mask(np.random.random(X.shape) < missing_values)
        
    if id_cols:
        for total_id_col in range(id_cols):
            id_col = [uuid.uuid4().hex for k in range(len(X))]
            X[f'id_{total_id_col}'] = id_col
    return X, cat_columns