# AdaptiveHybridRS
An adaptive hybrid recommender system using kernel density estimation to help resolve the cold start problem.

Our model pioneers the use of kernel density estimation in content-based filtering to produce item recommendation on a property dataset. We also construct a collaborative filtering recommender system, applying a variety of similarity metrics, and combine them in a hybrid model. Our adaptive recommender system is able to dynamically adjust the component models according to the level of data sparsity encountered.

The Python classes and their functions are relatively self-explanatory. The 'scripts' folder contains the range of experiments that were conducted for evaluation of our recommendation models.

- Python 3.6.1
- numpy 1.12
- pandas 0.20.1
- matplotlib 2.0.2
