#!/usr/bin/env python3
# borrowed from https://github.com/awslabs/amazon-denseclus/blob/main/denseclus/

import logging
import warnings

import numpy as np
import pandas as pd
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