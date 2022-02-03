from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

from .feats import *


import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from dython.nominal import associations
import matplotlib.pyplot as plt
from matplotlib import style

import seaborn as sns
import uuid
plt.rcParams["figure.figsize"]=20,20



def normalize_df(df, method):
    
    if method       == 'minmax':
        df = MinMaxScaler().fit_transform(df)
    elif method     == 'standard':
        df = StandardScaler().fit_transform(df)
    elif method     == 'power':
        df = PowerTransformer().fit_transform(df)
    return df


def al(df):
    """check any null value """
    return df.isnull().values.any()


def k_proto_data(df, cat_columns):
    
    df_train = cobj(df, cat_columns)
    cat_num  = [m for m,n in enumerate(df.dtypes) if n == 'object']
    return df_train, cat_num


def get_dummy_data(n_samples, 
                   columns, 
                   cat_col,
                   centers        = 3,
                   missing_values = None, 
                   id_cols        = None):
    
    """generate dummy mix data types df """
    
    X, y          = make_blobs(n_samples=n_samples, centers= centers, n_features=columns, random_state=12)
    
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
    return X, cat_columns, con_feats, y






#-------- new preprocessing -------



# preprocessing -> mixing value check, imputation
# Feature enginnering -> feature importance, feature aug, creating more features
# model selection
# explanation

def value_to_float(x):
  '''
  convert 95.5 to numeric value
  '''
  if type(x) == float or type(x) == int:
        return x



def sanity_checks(df, column_names):
    
    '''
    few columns which should not 
    contain any nan value
    '''
    
    for col_name in column_names:
        df = df[df[col_name].notna()]
    return df



# Standardizing all the numerical variables
def normalize_feat(df, con_columns, method):
    
    con_columns     = [col for col in df.columns if col in con_columns]
    print(f'normalizing selected columns -- {",".join(con_columns)} -- ')
    if method       == 'minmax':
        df[con_columns] = preprocessing.MinMaxScaler().fit_transform(df[con_columns])
        
    elif method     == 'standard':
        df[con_columns] = preprocessing.StandardScaler().fit_transform(df[con_columns])

    return df


def encode_columns(df, columns = False):
    
    '''encode columns for 
    categorical columns
    '''
        
    if columns:
        for col in columns:
            print('encoding..')
            print(col, df[col].nunique())
            l_enc   = LabelEncoder()
            df[col] = l_enc.fit_transform(df[col].values)
        
    else:
        for col in df.columns:
            if df[col].dtype == np.dtype('O'):
                print('encoding..')
                print(col, df[col].nunique())
                l_enc   = LabelEncoder()
                df[col] = l_enc.fit_transform(df[col].values)
    return df
    
    
    
def cobj(df, columns):
    
    for col_name in columns:
        df[col_name] = df[col_name].astype(object)
    return df


def missing_values_table(df):
    
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"   
           + "total values " + str(len(df)) +  
        " There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    mis_val_table_ren_columns['col_name'] = mis_val_table_ren_columns.index
    return mis_val_table_ren_columns



def trimm_correlated(df_in, cat_columns, 
                     threshold):
    
    df_corr           = associations(df_in, 
                                     nominal_columns = cat_columns)['corr']
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
    un_corr_idx       = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out            = df_in[un_corr_idx]
    return df_out


def get_types(df):
    
    id_columns = [k for k in df.columns if 'id' in str(k).lower()]
    cat_columns= df.select_dtypes(include=['object'])
    cat_columns= [col for col in cat_columns if col not in id_columns]
    con_columns= df.select_dtypes(include=['int','float'])
    con_columns= [col for col in con_columns if col not in id_columns]
    remain_cols= [col for col in df.columns if col not in id_columns 
                  if col not in cat_columns 
                  if col not in con_columns]
    
    if remain_cols:
        return id_columns, cat_columns, con_columns, remain_cols
    else:
        return id_columns, cat_columns, con_columns



def column_checks(df):
    
    counts = dict(df.nunique())
    to_del = [i for i,v in counts.items() if v == 1]
    
    if to_del:
        print(f'deleting columns -- {",".join(to_del)} -- Reason : Contain a Single Value')
        df     = df.drop(to_del, axis=1, inplace=False)
        print('before duplicate size', df.shape)
        df     = df.drop_duplicates()
        print('after duplicate removal', df.shape)
    

    return df


def del_nan_columns(df, 
                    missing_df,
                    missing_value = 2.0):
    
    
    if missing_value == 0.0 or missing_value == 0:
        print('dropping all nan values rows in any column..')
        df = df.dropna()
    else:
        col_to_delete = list(missing_df[missing_df['% of Total Values'] > missing_value].index)
        if col_to_delete:
            df            = df.drop(col_to_delete, axis=1, inplace = False)
            print(f'dropping columns -- {",".join(col_to_delete)} -- Reason : Contains a lot NaN values')
    return df


def convert_type(df_train):
    
    for y in df_train.columns:
        if(df_train[y].dtype == np.float64):
            df_train[y] = df_train[y].astype(int)
    return df_train


def uv(df):
    
    """ print unique values in df format"""

    df_values         = pd.DataFrame(df.nunique())
    return df_values


class DataFrameImputer(TransformerMixin):
    
    def __init__(self):
        
        """Impute missing values.
        """
        
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') 
                               else X[c].interpolate(limit_direction="both", 
                                                     method='linear') for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    


def autopreprocessing(df, 
                      cat_columns, 
                      id_columns      = None,
                      con_colmns      = None,
                      y               = None, 
                      allowed_missing = 23.0,
                      corr_thr        = 0.9):
    
    if id_columns:
        id_frame        = df[id_columns]
        
    
    """ ----- Data columns selection -------"""
    
    allowed_missing     = float(allowed_missing)
    total_columns       = list(cat_columns) + list(con_colmns)
    df                  = df[list(set(total_columns))]
    print(f'total_columns..')
    print(" ".join(total_columns))
    print(df.shape)
    
    """ ------------------------------------ """
    
    
    if y:
        print("Sanity checks..", df.shape)
        df              = sanity_checks(df, y)
    
    
    print('converting cat columns into object types..\n', df.shape)
    df                  = cobj(df, cat_columns)

    print('calculating missing values..\n', df.shape)
    mrf                 = missing_values_table(df)
    
    print(mrf)
    print('\n')
    
    df                 = del_nan_columns(df, mrf, 
                                          allowed_missing)
    df                 = column_checks(df)
    
    if allowed_missing!=0.0:
        print('imputing values..', df.shape)
        df              = DataFrameImputer().fit_transform(df)
    
    
    print('encoding cat columns', df.shape)
    final_cat_col   = [col_name for col_name in df.columns if col_name in cat_columns]
    print(f'Final cat features {",".join(final_cat_col)}')
    df              = cobj(df, final_cat_col)
    df_raw          = df.copy(deep=True)
    df              = encode_columns(df)
    
    

    """ 
    https://stats.stackexchange.com/questions/
    428187/what-should-be-done-first-handling-missing-data-or
    -finding-correlation-between
    
    """
    
    """ -----------------corr------------------------------------"""
    
    df_copy          = df.copy(deep=True)
    uio_corr_mat     = convert_type(df_copy)
    corr_mat         = trimm_correlated(uio_corr_mat, final_cat_col, 
                                         threshold = corr_thr)
    if corr_thr:
        df        = df[corr_mat.columns]
        
    """ ----------------------------------------------------------"""
    
    
    if id_columns:
        df     = df.merge(id_frame, left_index=True, right_index=True, how='inner')
        df_raw = df_raw.merge(id_frame, left_index=True, right_index=True, how='inner')
    
    
    """ Presering the data types """
    
    final_cat_col   = [col_name for col_name in df.columns if col_name in cat_columns]
    print(f'Final cat features {",".join(final_cat_col)}')
    df              = cobj(df, final_cat_col)
    df_raw          = df_raw[df.columns]
    df_raw          = cobj(df_raw, final_cat_col)
    
    return df, df_raw










# use case

# id_columns = ['usubjid', 'SUBJID', 'SITEID']
# cat_co     = ['LBTEST', 'LBNRIND', 'VISIT', 'SEX', 'RACE', 'ETHNIC', 'ACTARM', 'DTHDTC', 'DTHFL', 'COUNTRY']
# num_col    = ['LBSTRESN', 'VISITNUM', 'VISITDY', 'LBDTC', 'LBDY', 'AGE']



# er = autopreprocessing(df, 
#                        cat_columns    = cat_co, 
#                       id_columns      = id_columns,
#                       con_colmns      = num_col,
#                       y               = None,
#                       allowed_missing = 23.0, 
#                        corr_thr       = 0.8)