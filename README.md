# Anomaly Detection on UAVs 
<p align="center">
    <img src="gui/images/anomaly_detection_logo.png">
</p>
System's main goal is to create machine learning models for anomaly detection on UAVs.
The system allows creation and loading of machine learning models by using dynamic inputs. <br/><br/>
Moreover, the system displays different output plots and evaluation metrics which compare between different models and the diagnosis of anomalies which were found.
Running the system with dynamic parameters will allow us to extract many different machine learning models.
Comparing them based on different evaluation metrics will lead to obtaining the best machine learning models for anomaly detection.<br/><br/>
Those models will be used as **a baseline for a real-time & light-weight anomaly detection algorithm based on streaming data from UAV sensors
in to order to get the earliest possible detection of GPS spoofing attacks on UAV’s**.  

### Prerequisites

You should run the command (before run the system) in the console:

```
pip install -r requirements.txt
```


## Getting Started

First, you should clone the project to your local environment:

```
git clone https://github.com/liorpizman/AnomalyDetection.git
```

<br/>

Run 'guiController.py' file in order to run the system.
<img src="utils/images/shared/guiController.JPG"><br/>

Choose Between two different option:
<img height=300 width=300 src="utils/images/shared/mainWindow.JPG"><br/>

## Create new machine learning model

## Load existing machine learning model

**2.** Currently the flow of creating new model is working(load existing model will be available later).<br/>
**3.** Enter valid inputs for train,test and results directories.<br/>
**4.** Choose LSTM and edit the parameters in order to continue to next step in the system (LSTM is working, more algorithms will be available later).<br/>
**5.** Skip the next step (will be available later).<br/>
**6.** Choose cosine similarity function (optional).<br/>
**7.** Run new model.<br/>


## Generated Machine Learning Models 

* LSTM - Long Short-Term Memory
* SVR - Support Vector Regression
* Random Forest
* Multivariate Linear Regression


| Algorithm | Description |
| -- | -- |
| LSTM | An artificial recurrent neural network (RNN) architecture used in the field of deep learning. |
| SVR | A popular machine learning tool for classification and regression. |
| Random Forest | Are supervised ensemble-learning models used for classification and regression. |
| Multivariate Linear Regression | An approach for statistical learning. As the name implies, multivariate regression is a technique that estimates a single regression model with more than one outcome variable. |

## Train & Test Explained

| Data Set | Description |
| -- | -- |
| Train Set | Records containing sensors' values ​​for non-anomalous drone flights. |
| Test Set | Records containing sensors' values ​​for flights that have been attacked in various predefined attacks. |

## GPS Spoofing Attacks - ADS-B Data Sets

| Attack | Description |
| -- | -- |
| Up attack | Try crushing the drone by changing his height sensor data in the dataset. |
| Down attack | An attempt to raise the drone up and get him out of his real route. |
| Fore attack | Randomly change sensors’ values. |
| Random attack | Injection of real sensors’ data from another flight to current flight. |

## GPS Spoofing Attacks - Simulated Data Sets

| Attack | Description |
| -- | -- |
| Constant attack | Constant height and constant velocity. |
| Changing height attack | Constant height and changing velocity. |
| Changing velocity attack | Changing height and constant velocity. |
| Mixed attack | Changing height and changing velocity. |

## LSTM - Results Example 

---- to do ----

## SVR - Results Example 

---- to do ----

## Random Forest - Results Example 

---- to do ----

## Multivariate Linear Regression - Results Example 

---- to do ----

## Research Risks

* **Imbalanced data sets** - the amount of data about attacks is very small compared to drone's regular behavior data.
* **Duration of the attack detection** - the true detection rate of GPS attacks will be high (TPR) but the duration of the attack detection will be long so the drone will be abducted even though the attack was detected.
* **Results expectations** – machine learning models results can be different from our initial expectations.


## Python Libraries We Used

* [Keras](https://keras.io/) - the Python Deep Learning library.
* [Scikit-learn](https://scikit-learn.org/) - is an open source machine learning library that supports supervised and unsupervised learning.

## Data Sets

* **ADS-B data sets** - automatic dependent surveillance – broadcast - data sets
* **Simulated data sets** - data sets which are generated by a simulator

## Built With

* [PyCharm](https://www.jetbrains.com/pycharm/) - the Python IDE for Professional Developers
* [Flightradar24](https://www.flightradar24.com/) - ADS-B data sets

## Authors

* **Lior Pizman** - *Final project, SISE, BGU* - [Github](https://github.com/liorpizman/)
* **Yehuda Pashay** - *Final project, SISE, BGU* - [Github](https://github.com/yehudapashay)

See also the list of [contributors](https://github.com/liorpizman/AnomalyDetection/contributors) who participated in this project.

