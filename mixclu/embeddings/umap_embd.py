#!/usr/bin/env python3
# borrowed from https://github.com/awslabs/amazon-denseclus/blob/main/denseclus/

import logging
import warnings
import umap.umap_ as umap
from sklearn.base import BaseEstimator, ClassifierMixin
logger = logging.getLogger("denseclus")
logger.setLevel(logging.ERROR)
sh = logging.StreamHandler()
sh.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger.addHandler(sh)
from .utils import check_is_df, extract_categorical, extract_numerical

from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import DistanceMetric
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import preprocessing
import gower


from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn_extra.cluster import KMedoids

import umap

import io
import math
import pandas as pd
import numpy as np
import math

from community import community_louvain
import networkx as nx

from matplotlib import gridspec
import matplotlib.pyplot as plt



class Umap_embeddings(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        random_state: int = None,
        n_neighbors: int = 30,
        n_components: int = None,
        umap_combine_method: str = "intersection",
        verbose: bool = False,
    ):

        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.umap_combine_method = umap_combine_method

        if verbose:
            logger.setLevel(logging.DEBUG)
            self.verbose = True
        else:
            logger.setLevel(logging.ERROR)
            self.verbose = False
            # supress deprecation warnings
            # see: https://stackoverflow.com/questions/54379418

            def noop(*args, **kargs):
                pass

            warnings.warn = noop

        if isinstance(random_state, int):
            np.random.seed(seed=random_state)
        else:
            logger.info("No random seed passed, running UMAP in Numba")

    def __repr__(self):
        return str(self.__dict__)

    def fit(self, df: pd.DataFrame) -> None:
        """Fit function for call UMAP and HDBSCAN
        Parameters
        ----------
            df : pandas DataFrame
                DataFrame object with named columns of categorical and numerics
        Returns
        -------
            Fitted: None
                Fitted UMAPs and HDBSCAN
        """

        check_is_df(df)

        if not isinstance(self.n_components, int):
            self.n_components = int(round(np.log(df.shape[1])))

        logger.info("Extracting categorical features")
        self.categorical_ = extract_categorical(df)

        logger.info("Extracting numerical features")
        self.numerical_ = extract_numerical(df)

        logger.info("Fitting categorical UMAP")
        self._fit_categorical()

        logger.info("Fitting numerical UMAP")
        self._fit_numerical()

        logger.info("Mapping/Combining Embeddings")
        self._umap_embeddings()

    def _fit_numerical(self):
        numerical_umap = umap.UMAP(
            metric="l2",
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            random_state=self.random_state,
        ).fit(self.numerical_)
        self.numerical_umap_ = numerical_umap
        return self

    def _fit_categorical(self):
        categorical_umap = umap.UMAP(
            metric="dice",
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            random_state=self.random_state,
        ).fit(self.categorical_)
        self.categorical_umap_ = categorical_umap
        return self

    def _umap_embeddings(self):

        if self.umap_combine_method == "intersection":
            self.mapper_ = self.numerical_umap_ * self.categorical_umap_

        elif self.umap_combine_method == "union":
            self.mapper_ = self.numerical_umap_ + self.categorical_umap_

        elif self.umap_combine_method == "contrast":
            self.mapper_ = self.numerical_umap_ - self.categorical_umap_

        elif self.umap_combine_method == "intersection_union_mapper":
            intersection_mapper = umap.UMAP(
                random_state=self.random_state,
                n_neighbors=self.n_neighbors,
                n_components=self.n_components,
                min_dist=0.0,
            ).fit(self.numerical_)
            self.mapper_ = intersection_mapper * (
                self.numerical_umap_ + self.categorical_umap_
            )

        else:
            raise KeyError("Select valid  UMAP combine method")

        return self




def get_umap_embeddings(df,  
                        n_components, 
                        umap_combine_method, 
                        random_state = 42, 
                        n_neighbors  = 30):

  dat_ = Umap_embeddings(random_state        = random_state, 
                         n_neighbors         = n_neighbors,
                         n_components        = n_components,
                         umap_combine_method = umap_combine_method)
  dat_.fit(df)
  return dat_.mapper_.embedding_


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



def evaluate_clusters(scores,  preds, labels, name='', X=None):
    
    if X is not None:

        silhouette = silhouette_score(X, preds, metric='euclidean')
        cal_har = calinski_harabasz_score(X, preds)
        dav_bould = davies_bouldin_score(X, preds)

        adj_mut_info = adjusted_mutual_info_score(labels, preds, average_method='arithmetic')
        adj_rand = adjusted_rand_score(labels, preds)

        content = {'Algorithm':name,
                   'Silhouette \u2191 ':silhouette,
                   'Calinski_Harabasz \u2191 ':cal_har,
                   'Davis Bouldin \u2193 ':dav_bould,
                   'Adjusted_Mutual_Info \u2191 ':adj_mut_info,
                   'Adjusted_Rand_Score \u2191 ':adj_rand}

        scores = scores.append(content, ignore_index = True)

    else:

        adj_mut_info = adjusted_mutual_info_score(labels, preds, average_method='arithmetic')
        adj_rand = adjusted_rand_score(labels, preds)

        content = {'Algorithm':name,
                   'Silhouette \u2191 ':np.NaN,
                   'Calinski_Harabasz \u2191 ':np.NaN,
                   'Davis Bouldin \u2193 ':np.NaN,
                   'Adjusted_Mutual_Info \u2191 ':adj_mut_info,
                   'Adjusted_Rand_Score \u2191 ':adj_rand}

        scores = scores.append(content, ignore_index = True)
    return scores



def calculate_gower_distance(df, cat_columns = None):
    
    cat_columns = list(cat_columns)
    if cat_columns:
        variable_distances = gower.gower_matrix(df,cat_features= 
                           [True if df[k].dtypes == np.object else False 
                            for k in df.columns])
    else:
        variable_distances = gower.gower_matrix(df)
    
    variable_distances[np.isnan(variable_distances)] = 0
    return variable_distances


def calculate_zscore(df, columns):
    '''
    scales columns in dataframe using z-score
    '''
    df = df.copy()
    for col in columns:
        df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

    return df



def one_hot_encode(df, columns):
    
    '''
    one hot encodes list of columns and
    concatenates them to the original df
    '''

    concat_df = pd.concat([pd.get_dummies(df[col], drop_first=True, prefix=col) for col in columns], axis=1)
    one_hot_cols = concat_df.columns

    return concat_df, one_hot_cols



def normalize_column_modality(df, columns):
    '''
    divides each column by the probability μₘ of the modality 
    (number of ones in the column divided by N) only for one hot columns
    '''

    length = len(df)
    for col in columns:

        weight = math.sqrt(sum(df[col])/length)
        df[col] = df[col]/weight

    return df



def center_columns(df, columns):
    '''
    center columns by subtracting the mean value
    '''
    for col in columns:
        df[col] = (df[col] - df[col].mean())
    return df



def FAMD_2(df, n_components=2):
    '''
    Factorial Analysis of Mixed Data (FAMD), 
    which generalizes the Principal Component Analysis (PCA) 
    algorithm to datasets containing numerical and categorical variables

    a) For the numerical variables
    - Standard scale (= get the z-score)

    b) For the categorical variables:
    - Get the one-hot encoded columns
    - Divide each column by the square root of its probability sqrt(μₘ)
    - Center the columns

    c) Apply a PCA algorithm over the table obtained!

    '''

    variable_distances = []
    numeric_cols = data.select_dtypes(include=np.number)
    cat_cols = data.select_dtypes(include='object')

    # numeric process
    normalized_df = calculate_zscore(df, numeric_cols)
    normalized_df = normalized_df[numeric_cols.columns]

    # categorical process
    cat_one_hot_df, one_hot_cols = one_hot_encode(df, cat_cols)
    cat_one_hot_norm_df = normalize_column_modality(cat_one_hot_df, one_hot_cols)
    cat_one_hot_norm_center_df = center_columns(cat_one_hot_norm_df, one_hot_cols)

    # Merge DataFrames
    processed_df = pd.concat([normalized_df, cat_one_hot_norm_center_df], axis=1)

    # Perform (PCA)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(processed_df)

    return principalComponents


# use Umap to do embedding then cluster on that
def umap_reduce(df, intersection=False):
    
    numerical = df.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
      
    ##preprocessing categorical
    categorical = df.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)



    #Embedding numerical & categorical
    fit1 = umap.UMAP(random_state=12).fit(numerical)
    fit2 = umap.UMAP(metric='dice', n_neighbors=250).fit(categorical)

    numeric_embedding = fit1.embedding_
    numeric = pd.DataFrame(
                         {'x': numeric_embedding[:,0],
                         'y':  numeric_embedding[:,1],
                        })


    categorical_embedding = fit2.embedding_
    categorical = pd.DataFrame(
                         {'x': categorical_embedding[:,0],
                         'y':  categorical_embedding[:,1],
                        })


    # intersection will resemble the numerical embedding more.
    if intersection:
        embedding = fit1 * fit2

    # union will resemble the categorical embedding more.
    else:
        embedding = fit1 + fit2

    umap_embedding = embedding.embedding_


    results = pd.DataFrame(
                        {'x': umap_embedding[:,0],
                         'y':  umap_embedding[:,1],
                        })
    
    return results, umap_embedding




def elbow_method_kmeans(df, space=(2,11)):

    cost = []
    n_clusters = []

    start = space[0]
    stop  = space[1]
    for k in range(start, stop):
        kmeans = KMeans(n_clusters=k, verbose=0)
        kmeans.fit(df)
        cost.append(kmeans.inertia_)
        n_clusters.append(k)


    results = pd.DataFrame(
                        {'n_clusters': n_clusters,
                        'cost': cost,
                        })
  
    return (p9.ggplot(results, p9.aes(x='n_clusters', y='cost'))
            + p9.geom_point()
            + p9.geom_line()
            + p9.ggtitle('Elbow Plot'))




def get_knn_bins(df, cols, 
                 bins=5, 
                 drop_cols=True, 
                 encode = True):
    
    k_columns = []
    
    for col in cols:
    
        kmeans  = KMeans(n_clusters=bins).fit(df[col].to_frame().values.reshape(-1,1))
        results = pd.DataFrame(kmeans.labels_, columns=[col + '_centroid'])

        df = df.reset_index()
        df[col + '_centroid'] = results[col + '_centroid']

        knn_bin_df = pd.DataFrame(kmeans.cluster_centers_)
        knn_bin_df = knn_bin_df.astype(int).reset_index()

        temp_df = pd.merge(df[col + '_centroid'],
                           knn_bin_df, 
                           left_on=col + '_centroid',
                           right_on='index',
                           how='left')

        # rename empty column header 0 -> column_name value
        temp_df = temp_df.rename(columns={0:col+'_value'})

        temp_df.loc[:,col+'_value'] = col + '_' + temp_df[col+'_value'].astype(str)

        df = pd.concat([df, temp_df[col+'_value']], axis=1)
        df.drop([col + '_centroid', 'index'], axis=1, inplace=True)
        k_columns.append(col+'_value')
    
    cat_columns = [k for k in df.columns if k not in cols]
    print("New cat columns ", ",".join(k_columns))
    df = cobj(df, cat_columns)
    
    if drop_cols:
        df = df.drop(cols, axis=1)
    
    if encode:
        df = encode_columns(df, k_columns)

    return df



def convert_df_to_sgraph_network(df):
    '''
    This function converts a dataframe into an edge list and finally
    into a network graph
    '''
    df = df.copy()
    edges_df = pd.DataFrame()
    # create a name for each row
    length = len(df)
    row_names = ['row '+ str(i) for i in range(1, length+1)]

    original_cols = df.columns
    df['row_name'] = row_names

    for col in original_cols:
        col_edge_df = df[['row_name', col]].rename(columns={col:'to'})
        edges_df = pd.concat([edges_df, col_edge_df], axis=0)

    # set the edge weights to one
    edges_df['weight'] = 1
    edges_df = edges_df.groupby(['row_name', 'to']).count().reset_index()
    edges_df.rename(columns={'row_name':'from'}, inplace=True)

    graph = nx.from_pandas_edgelist(edges_df, source='from',
                                  target='to', edge_attr=['weight'])
  
    return graph