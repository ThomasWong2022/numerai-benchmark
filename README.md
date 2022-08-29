# numerai-benchmark

Benchmark Models for Numerai competition 

Python package to benchmark machine learning models for analysing non-stationary data. Both tabular data and time-series data are supported. 

The package consists of a general module which is applicable for time-series and other non-stationary datasets and a finance module with useful functions to perform workflow in finance. 

There are already plenty of well-maintained time-series packages (Ex. prophet(Meta)) which works well for stationary data and some well-maintained package for quantitative trading, which requires users to have a good knowledge in finance. We aim to provide a package which address the gap, a multi-purpose prediction package for different datasets, which can be time-series and tabular data. 


## Why another package for finance/time-series

Alpalens(Quantopian), Lean(QuantConnect), qlib(Microsoft) are packages that receive more than 3k stars on Github (Data as of 28 Nov 2021) for quantitative trading. We list serveral different limitations of these package 

- Alphalens/Zipline(Quantopian)
    - Not updated for at least 2 years since the shutdown of Quantopian website
    - Zipline does not support the latest Python version and pandas 
    
- Lean(QuantConnect)
    - Written in C#, it is not possible for Python developers to add on extra functions easily
    - Unlike Alphalens which allows users to quickly test a trading idea without doing a full backtest, Lean does not corresponding functinality for data exploration

- qlib(Microsoft)
   - Users need to prepare a database of financial data and specify various parameters before running a backtest
   - Not general purpose as many assumptions for quantitative trading is embedded in the system, even it supports many machine learning models which are useful for researchers outside finance


We decided to develop a lightweight Python package which allows users to quickly benchmark existing prediction models for time-series and other non-stationary datasets. We provide standard recipes built on scikit-learn and other Python packages which allows users to run benchmarking studies with as few lines as possible. This enables researchers who are not specialised in machine learning (such as medical research) to perform analysis on their experimental datasets with the latest models with minimal effort. The recipes who also serves as good baselines for machine learning researchers to compare models they developed with. 

Another purpose for this package is introduce methods commonly used in quantitative finance in other scientific forecast/prediction problems. In particular, reformulate the online/stream forecast problem as a trading system where the agent receives reward according to the correctness of forecasts. Under this new framework, self-driving cars and trading systems can be solved under the common framework of achieving a certain goal while minimising risks. 




## Target use case

- Finance Research: Researchers and students can use this package to perform efficient analysis of financial data from WRDS, which is a popular data source for finance research
- Scientific Research: Researchers in other areas such as medicine and engineering will also find the package useful for performing different forecasting tasks in theire relevant fields. 
- Data Scientists: Data scientists will find some of the receipes useful for Numerai and Kaggle competitions. 



## Design principles 


- Academic-driven: We aim to incoporate the latest research in machine learning, especially deep learning and reinforcement learning into our benchmark models. We decided not to include older models such as Wavelets and Dynamic Linear Models as they are no longer considered useful for (financial) forecasting. For similar results, we decided not to support technical analysis in trading. 
- Lightweight: The minimal amount of data that a user needs to use our package is a Dataframe of features and a Series of target   
- Data Agnostic: The core functions are written with minimal assumptions on data, unlike other Python package such as qlib which requires users to build a database for financial data. 
- General Purpose: The core functions to benchmark prediction models is written without explicit references to financial data unlike most quantitative trading package
- Comptability: We aim to keep most of the code compatible with scikit-learn, which is the gold standard for machine learning in Python. 




### Proposed workflow for non-stationary data 

Step 1: Data Preparation and Cleaning
    - Download datasets from various data sources
    - Process and clean data 
    - Create Prediction targets, absolute/relative returns or quantiles 
Step 2: Data Diagnostic
    - Run data diagnostic tests
Step 3: Feature Engineering
    - Create features for time-series dataset
    - Perform feature selection on generated features
Step 4: Select Prediction Model
    - Tabular model on transformed datasets of time-series
    - Deep Learning Model on 2D or higher dimension of time-series data 
Step 5: Hyper-parameter optimisation
    - Create hyper-parameter search space  
    - Choose parameter search algorithm (Grid Search/Random Search/Bayesian Optimization) 
Step 6: Train model 
    - Train model with optimised hyper-parameters
    - Generate predictions on test dataset
    - Evaluate Model Performance 




## Related packages (machine learning)

- scikit-learn: The gold standard package for Machine Learning in Python
- scikit-multiflow: A scikit-learn compatible package for stream learning
- sktime: A scikit-learn compatible package for time-series forecasting
- xgboost: distributed gradient boosting library 
- lightgbm: gradient boosting framework by Microsoft
- CatBoost: gradient boosting framework by Yandex
- TabNet: Attentive Interpretable Tabular Learning in PyTorch
- OpenAI-Gym: Creating standardised reinforcemet learning environment
- ray-Tune: Scalable Hyperparameter Tuning
- ray-Rllib: Scalable Reinforcement Learning


## Related packages (natural langauge processing)
- Huggingface-Transformers: State-of-the-art Natural Language Processing for Pytorch, TensorFlow, and JAX
- TextBlob: 




## Related packages (finance)
- qlib: A Python package to perform backtesting sponsored by Microsoft
- pyfinance: A Python package to download data from Yahoo Finance which is widely used by retail traders
- investpy: A Python package to download data from investing.com
- pyportfolioopt: A lightweight Python package to optimise portfolio 




## Development Roadmap 

The project is currently at Stage 1 which is to build a basic benchmarking framework. We aim to keep all the code in Stage 1 open-source. Stage 2 and 3 would consists of scaling up the system for paper and live trading. Due to resources limitations, there is no concrete plans in bringing project to Stage 2 and 3. Industry sponsors and retail traders who are interested in taking the project to Stage 2 and 3 can contact Thomas Wong (@ThomasWong2022) for more details.  Due to the competitaive nature of quantitative trading, code developed in Stage 2 and 3 might not be open-source depending on the contract agreement with relevant sponsors. 



- Stage 1: Basic functions  
    - Develop utilities functions to process (financial) time-series
    - Develop workflow for benchmarking time-series prediction models
    - Develop basic trading strategies using option replication and butterfly model
    - Develop Smart-beta factors 
    - Develop basic transaction costs and model 
    - Develop a basic backtesting framework using OpenAI gym
    - Develop basic data classes for Yahoo Finance, CRSP and Compustat data
    - Provide standard benchmark recipes for trading (technical analysis/machine learning)
    

- Stage 2: Advanced functions 
    - Integrate financial and filings data from SEC EDGAR
    - Build a better volatility model
    - Build a more realistic transaction costs and backtesting framework for different trading systems 
    - Build a risk management and hedging model 
    - Build a dynamic portfolio optimisation model
    - Improve existing infrastructure for parallel computing 
    - Build interface to support common databases for data storage (Ex. Amazon DynamoDB, 
    - Build interface to support cloud computing (Ex. Docker, Amazon Serverless)
    
    
- Stage 3: Paper/Live Trading support
    - Web Application and Dashboards 
    - Support for historical/live data from different data sources (Ex. Nasdaq data link/IEX)
    - Support for international finanical data
    - Support for company research reports and transcripts 
    - Support for news and sentiment data 
    - Support paper trading systems (Ex. Alpaca)
    - Support live trading systems 



## Acknowledgement 

Core Developers: Thomas Wong

Developers:

We would also like to thank the following for their helpful discussions:

