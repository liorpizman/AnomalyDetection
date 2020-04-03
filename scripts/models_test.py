import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor


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
    pass


path = "C:\\Users\\Yehuda Pashay\\Desktop\\fligth_data\\data_set\\train\\chicago_to_guadalajara"

# run_logistic_regression(path)
run_isolation_forest(path)
