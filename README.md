The code is substantially improved in the new autoML tool THOR described below. If you want to know more about THOR please contact Thomas Wong by email mw4315@ic.ac.uk


# THOR: Time-Varying High-dimensional Ordinal Regression 

THOR is a new autoML tool for temporal tabular datasets and time series. It handles high dimensional datasets with distribution shifts better than other tools. Inspired by the Numerai competiton, THOR has evolved from a specific tool for Numerai competition into a general ML pipeline which has many applications in finance and healthcare. 

In the following I list some features in THOR. It is not an exhausive list and there are more proprietary features that is not listed here. 

## GBDT2.0

A customisted LightGBM-based Gradient Boosting Decision Trees models for temporal tabular datasets. 

## DeepLearner2.0

A novel deep learning model for temporal tabular datasets, which complements well with the above GBDT-based models. 


## PortfolioOpt2.0

A new method to combine predictions from machine learning model using well-known theories from finance.
Using the best research methods from **both** finance and reinfrocement learning, 
the method can maximise the portfolio return (or minimise the given loss function) within required risk metrics.


## TimeSeriesHybrid 

A new method which combines classical and machine learning techniques for feature engineering and sequence modelling. 
A hybrid approach which demonstrate robust performances for high dimensional time-series. 

## TrendFollower2.0 

An enhanced implmentation of trend following strategies with improved robustness and lower risks than the standard implmentation of moving averages. 





## Citation
If you are using this package in your scientific work, we would appreciate citations to the following preprint on arxiv.

[Robust machine learning pipelines for trading market-neutral stock portfolios](https://arxiv.org/abs/2301.00790 )

Bibtex entry:
```
@misc{https://doi.org/10.48550/arxiv.2301.00790,
  doi = {10.48550/ARXIV.2301.00790},
  
  url = {https://arxiv.org/abs/2301.00790},
  
  author = {Wong, Thomas and Barahona, Mauricio},
  
  keywords = {Computational Finance (q-fin.CP), Computational Engineering, Finance, and Science (cs.CE), Machine Learning (cs.LG), FOS: Economics and business, FOS: Economics and business, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Robust machine learning pipelines for trading market-neutral stock portfolios},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```









