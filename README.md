<h1 align="center">Mixclu </h1>

<p align="center">A Python package for unsupervised mix data types clustering </p>
<p align="center"> Contribute and Support </p>

Mixclu is a library for doing unsupervised mix data types clustering. This includes a variety of combination models including kmeans-onehot, gower distance, umap etc. The goal is to provide an easy to use implementation for each algorithm along with a scikit-learn API.


[![GitHub license](https://img.shields.io/badge/License-Creative%20Commons%20Attribution%204.0%20International-blue)](https://github.com/monk1337/Mixclu/blob/main/README.md)
[![GitHub commit](https://img.shields.io/github/last-commit/monk1337/Mixclu)](https://github.com/monk1337/Mixclu/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


 ## More features and suggestions are welcome.

- [**Check all models**](https://github.com/monk1337/Mixclu/blob/main/All_models_demo.ipynb)
- [**Auto Preprocessing**](https://github.com/monk1337/Mixclu/blob/main/notebooks/preprocessing.ipynb)

### Quick Start

```python
from mixclu import *

id_col, cat_columns, con_col = get_types(df)
""" define continuous and categorical columns"""

umap_kbins_emd, kbins_umap_model = categorical_embedding_model(df, 
                                                            bin_con_columns = con_col, 
                                                            no_of_clusters  = 4,
                                                            bin_bins        = 5)

```


```python

from mixclu import *

"""Kmeans one hot mix model"""

model_result = kmeans_onehot_mix(df, 
                               cat_columns, 
                               total_clusters = 4)
                               
```


```python

from mixclu import *

"""using gower distance as embeddings"""

clusters, gower_embeddings = kmoid_grower(df, 
                                           cat_columns, 
                                           no_of_clusters = 4)
                               
```



```python

from mixclu import *

"""(Mirkin, 2005, pp. 85–91)"""

z_score, model_result = Mirkin_model(df, 
                                     cat_columns, 
                                     no_of_clusters = 4)
                               
```



### Todo

- [ ] Autoencoder, Vae based clustering
- [ ] Tabformer, tabnet based embeddings then clustering
- [ ] Implement paper : [**Affinity Learning for Mixed Data Clustering**](https://www.ijcai.org/proceedings/2017/0302.pdf)
- [ ] Implement paper : [**A Multi-View Clustering for Mixed Data**](https://pure.hw.ac.uk/ws/portalfiles/portal/44173473/09347440.pdf)