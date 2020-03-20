# AnomalyDetection

A system which is able to create offline machine learning models for anomaly detection. The system allows creation and loading of machine learning models by using dynamic inputs. Moreover, the system displays different output plots and evaluation metrics which compare between different models and the diagnosis of anomalies which were found. Running the system with dynamic parameters will allow us to extract many different machine learning models. Comparing them based on different evaluation metrics will lead to obtaining the best machine learning models for anomaly detection. Those models will be used as a baseline for a real-time & light-weight anomaly detection algorithm based on streaming data from UAV sensors in to order to get the earliest possible detection of GPS spoofing attacks on UAV’s.  

## Getting Started

First, you should clone the project to your local computer.
1. Run 'guiController.py' file in order to run the system.
2. Currently the flow of creating new model is working(load existing model will be available later).
3. Enter valid inputs for train,test and results directories.
4. Choose LSTM and edit the parameters in order to continue to next step in the system (LSTM is working, more algorithms will be available later).
5. Skip the next step (will be available later).
6. Choose cosine similarity function (optional).
7. Run new model.

### Prerequisites

You should run the command (before run the system) in the console

```
pip install -r requirements.txt
```

## Built With

* [PyCharm](https://www.jetbrains.com/pycharm/) - The Python IDE for Professional Developers
* [Flightradar24](https://www.flightradar24.com/) -ADS-B data sets

## Authors

* **Lior Pizman** - *Prototype initial work* - [Github](https://github.com/liorpizman/)
* **Yehuda Pashay** - *Prototype initial work* - [Github](https://github.com/yehudapashay)

See also the list of [contributors](https://github.com/liorpizman/AnomalyDetection/contributors) who participated in this project.

