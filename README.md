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



### Package requirements (2022 March) 


Package                 Version   Location
----------------------- --------- ----------------------------------
absl-py                 1.0.0
aiohttp                 3.8.1
aiosignal               1.2.0
antlr4-python3-runtime  4.8
argon2-cffi             21.3.0
argon2-cffi-bindings    21.2.0
asttokens               2.0.5
async-timeout           4.0.2
attrs                   21.4.0
backcall                0.2.0
beautifulsoup4          4.10.0
black                   22.1.0
bleach                  4.1.0
blis                    0.7.5
cachetools              5.0.0
catalogue               2.0.6
catboost                1.0.4
category-encoders       2.2.2
certifi                 2021.10.8
cffi                    1.15.0
charset-normalizer      2.0.11
click                   8.0.3
cloudpickle             2.0.0
colorama                0.4.4
commonmark              0.9.1
configparser            5.2.0
cycler                  0.11.0
cymem                   2.0.6
dask                    2022.1.1
debugpy                 1.5.1
decorator               5.1.1
defusedxml              0.7.1
Deprecated              1.2.13
distributed             2022.1.1
docker-pycreds          0.4.0
einops                  0.3.0
entrypoints             0.4
esig                    0.9.7
executing               0.8.2
fastai                  2.5.3
fastcore                1.3.27
fastdownload            0.0.5
fastprogress            1.0.0
filelock                3.4.2
fonttools               4.29.1
frozenlist              1.3.0
fsspec                  2022.1.0
future                  0.18.2
gitdb                   4.0.9
GitPython               3.1.26
google-auth             2.6.0
google-auth-oauthlib    0.4.6
graphviz                0.19.1
grpcio                  1.43.0
HeapDict                1.0.1
huggingface-hub         0.4.0
hyperopt                0.2.7
idna                    3.3
iisignature             0.24
importlib-metadata      4.10.1
importlib-resources     5.4.0
ipykernel               6.8.0
ipython                 8.0.1
ipython-genutils        0.2.0
ipywidgets              7.6.5
jedi                    0.18.1
Jinja2                  3.0.3
joblib                  1.1.0
jsonschema              4.4.0
jupyter-client          7.1.2
jupyter-core            4.9.1
jupyterlab-pygments     0.1.2
jupyterlab-widgets      1.0.2
kiwisolver              1.3.2
langcodes               3.3.0
lightgbm                3.3.2
llvmlite                0.38.0
locket                  0.2.1
lxml                    4.7.1
Markdown                3.3.6
MarkupSafe              2.0.1
matplotlib              3.5.1
matplotlib-inline       0.1.3
matrixprofile           1.1.10
mistune                 0.8.4
msgpack                 1.0.3
multidict               6.0.2
multitasking            0.0.10
murmurhash              1.0.6
mypy-extensions         0.4.3
nbclient                0.5.10
nbconvert               6.4.1
nbformat                5.1.3
nest-asyncio            1.5.4
networkx                2.7.1
notebook                6.4.8
numba                   0.55.1
numerapi                2.9.4
numpy                   1.21.5
oauthlib                3.2.0
omegaconf               2.1.1
packaging               21.3
pandas                  1.1.5
pandocfilters           1.5.0
parso                   0.8.3
partd                   1.2.0
pathspec                0.9.0
pathy                   0.6.1
patsy                   0.5.2
pexpect                 4.8.0
pickleshare             0.7.5
Pillow                  9.0.1
pip                     21.2.4
platformdirs            2.4.1
plotly                  4.14.3
preshed                 3.0.6
prometheus-client       0.13.1
promise                 2.3
prompt-toolkit          3.0.26
protobuf                3.11.2
psutil                  5.9.0
ptyprocess              0.7.0
pure-eval               0.2.2
py4j                    0.10.9.5
pyarrow                 7.0.0
pyasn1                  0.4.8
pyasn1-modules          0.2.8
pycparser               2.21
pydantic                1.8.2
pyDeprecate             0.3.1
Pygments                2.11.2
pynndescent             0.5.6
pyparsing               3.0.7
pyrsistent              0.18.1
python-dateutil         2.8.2
pytorch-lightning       1.5.9
pytorch-tabnet          3.0.0
pytrend                 0.0.1     /disk3/thomas/factor_investing/src
pytz                    2021.3
PyYAML                  5.4.1
pyzmq                   22.3.0
readme-renderer         29.0
regex                   2022.1.18
requests                2.27.1
requests-oauthlib       1.3.1
requests-toolbelt       0.9.1
retrying                1.3.3
rfc3986                 1.4.0
rich                    11.1.0
rsa                     4.8
sacremoses              0.0.47
scikit-learn            1.0.2
scipy                   1.8.0
seaborn                 0.11.2
Send2Trash              1.8.0
sentry-sdk              1.5.4
setuptools              59.5.0
shortuuid               1.0.8
six                     1.16.0
sktime                  0.10.0
smart-open              5.2.1
smmap                   5.0.0
sortedcontainers        2.4.0
soupsieve               2.3.1
spacy                   3.2.1
spacy-legacy            3.0.8
spacy-loggers           1.0.1
srsly                   2.4.2
stack-data              0.1.4
statsmodels             0.13.1
stumpy                  1.10.2
subprocess32            3.5.4
tblib                   1.7.0
tensorboard             2.8.0
tensorboard-data-server 0.6.1
tensorboard-plugin-wit  1.8.1
terminado               0.13.1
testpath                0.5.0
thinc                   8.0.13
threadpoolctl           3.1.0
tokenizers              0.11.4
tomli                   2.0.0
toolz                   0.11.2
torch                   1.8.1
torchmetrics            0.7.1
tornado                 6.1
tqdm                    4.62.3
traitlets               5.1.1
transformers            4.16.2
tsfresh                 0.19.0
twine                   3.3.0
typer                   0.4.0
typing_extensions       4.0.1
umap-learn              0.5.2
urllib3                 1.26.8
wandb                   0.10.11
wasabi                  0.9.0
watchdog                2.1.6
wcwidth                 0.2.5
webencodings            0.5.1
Werkzeug                2.0.2
wheel                   0.37.1
widgetsnbextension      3.5.2
wrapt                   1.13.3
xgboost                 1.5.2
yarl                    1.7.2
yfinance                0.1.70
zict                    2.0.0
zipp                    3.7.0
