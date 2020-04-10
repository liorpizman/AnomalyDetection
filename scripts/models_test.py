import pandas as pd
from numpy import concatenate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX


def run_isolation_forest(file_path):
    features_list = ['Direction', 'Speed']
    df_train = pd.read_csv(f'{file_path}/without_anom.csv')

    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    random_model = MultiOutputRegressor(
        RandomForestRegressor(max_depth=2, max_features="sqrt")
    )

    # lab_enc = preprocessing.LabelEncoder()
    # training_scores_encoded = lab_enc.fit_transform(X_train)
    random_model.fit(X_train, X_train)
    pred = random_model.predict(X_train)
    # isolation_model = MultiOutputRegressor(IsolationForest()).fit(X_train)
    # pred = isolation_model.predict(X_train)
    test_path = "C:\\Users\\Yehuda Pashay\\Desktop\\fligth_data\\data_set\\test\\chicago_to_guadalajara\\down_attack"
    df_test = pd.read_csv(f'{test_path}/sensors_8.csv')
    df_test = df_test[features_list]

    Y_test = scalar.transform(df_test)
    test_pred = random_model.predict(Y_test)
    a = 4


def run_logistic_regression(file_path):
    df_train = pd.read_csv(f'{file_path}/without_anom.csv')
    features_list = ['Direction', 'Speed']
    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    logistic_model = LogisticRegression()

    # multi_model = MultiOutputRegressor(LogisticRegression())
    #
    # multi_model.fit(X_train, X_train)
    # multi_predict = multi_model.predict(X_train)

    logistic_model.fit(X_train, X_train)
    predict = logistic_model.predict(X_train)


def run_linear_regression(file_path):
    df_train = pd.read_csv(f'{file_path}/without_anom.csv')
    features_list = ['Direction', 'Speed']
    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    linear_model = LinearRegression()
    multi_model = MultiOutputRegressor(LinearRegression())

    linear_model.fit(X_train, X_train)
    multi_model.fit(X_train, X_train)

    linear_model_params = linear_model.get_params()
    multi_model_params = multi_model.get_params()

    print(linear_model_params)
    print(multi_model_params)


def run_MLP_model(file_path):
    df_train = pd.read_csv(f'{file_path}/without_anom.csv')
    features_list = ['Direction', 'Speed']
    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    model = MLPRegressor()
    model.fit(X_train, X_train)
    pred = model.predict(X_train)

    multi_model = MultiOutputRegressor(MLPRegressor())
    multi_model.fit(X_train, X_train)
    multi_pred = model.predict(X_train)


def run_ARIMA_model(train_file_path, test_file_path):
    df_train = pd.read_csv(f'{train_file_path}/without_anom.csv')
    features_list = ['third_dis', 'Direction', 'Speed']
    df_train = df_train[features_list]

    scalar = MaxAbsScaler()

    X_train = scalar.fit_transform(df_train)

    X_Direction_train = X_train[:, 0]

    Y_Direction_train = X_train[:, 0]

    history = [x for x in X_Direction_train]

    predictions = list()

    for t in range(len(Y_Direction_train)):
        model = ARIMA(history, order=(10, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat[0])
        obs = Y_Direction_train[t]
        history.append(obs)
        print('index: %d , predicted= %f , expected= %f , difference= %f' % (t, yhat, obs, abs(obs - yhat)))

    model_varmax(X_train)

    a = 1


def model_varmax(X_test):
    model = VARMAX(X_test, order=(1, 1))
    model_fit = model.fit(disp=-1)
    print(model_fit.summary().tables[1])
    predictions = model_fit.forecast(steps=10)
    print(predictions)


path = "C:\\Users\\Yehuda Pashay\\Desktop\\fligth_data\\data_set\\train\\chicago_to_guadalajara"

test_path = "C:\\Users\\Yehuda Pashay\\Desktop\\fligth_data\\data_set\\test\\chicago_to_guadalajara\\fore_attack"
# run_logistic_regression(path)
# run_MLP_model(path)
run_ARIMA_model(path, test_path)
