<h2 align="center">MixClu </h2>

<p align="center">A Python package for unsupervised mix data types clustering </p>
<p align="center"> Contribute and Support </p>


[![GitHub license](https://img.shields.io/badge/License-Creative%20Commons%20Attribution%204.0%20International-blue)](https://github.com/monk1337/Mixclu/blob/main/README.md)
[![GitHub commit](https://img.shields.io/github/last-commit/monk1337/Mixclu)](https://github.com/monk1337/Mixclu/commits/master)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


 ## More features and suggestions are welcome.


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
