from pathlib import Path
import pandas as pd
import numpy as np
from data_processing import PROCESSED_DATA_FILE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
#from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import max_error
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical


RESULTS_PATH = Path('results')
RESULTS_PATH.mkdir(exist_ok=True)

timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H_%M')
result_file_path = str(RESULTS_PATH / f'results_{timestamp}.csv')

TARGETS = ['Ra', 'Wz']

linear_param_space = {
    'model': [LinearRegression()],
    'n_jobs': -1,
    'copy_X': True,
}

svr_param_space = {
    'model': [SVR()],
    'model__kernel': Categorical(['linear', 'sigmoid']),
    'model__C': Real(1, 100),
    'model__epsilon': Real(0.1, 20),

}

rf_param_space = {
    'model': [RandomForestRegressor(random_state=42)],
    #'model__criterion': Categorical(['gini', 'entropy']),
    'model__n_estimators': Integer(50, 300),
    #'model__max_features': Integer(3, 27),
    'model__max_depth': Integer(2, 20),
    #'model__min_samples_split': Integer(2, 200),
    'model__min_samples_leaf': Integer(2, 200),
}

xgboost_param_space = {
    'model': [XGBRegressor(objective="reg:squarederror")],
    'model__learning_rate': Real(0.01, 0.9),
    'model__n_estimators': Integer(50, 500)

}

def train_models():
    df = pd.read_csv(PROCESSED_DATA_FILE)
    target_df = df[TARGETS]
    X = df.drop(columns=TARGETS)

    MODELS = [LinearRegression(), RandomForestRegressor(), SVR(), XGBRegressor()]
    results = []
    for target in TARGETS:
        y = target_df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)

        for param_space, n_iter in [
            #(linear_param_space, 5),
            (svr_param_space, 10),
            (rf_param_space, 30),
            (xgboost_param_space, 30)
        ]:
            model_name = param_space['model'][0].__class__.__name__
            print(target)
            print('\t' + model_name)
            pipe = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('model', None)
            ])
            opt = BayesSearchCV(
                pipe,
                search_spaces=[(param_space, n_iter)],
                cv=3,
                return_train_score=True,
                n_jobs=-1,
                refit=True,
                random_state=42,
                scoring='r2',
                verbose=0
            )

            np.int = int
            opt.fit(X_train, y_train)
            best_estimator = opt.best_estimator_

            y_pred_train = best_estimator.predict(X_train)
            y_pred_test = best_estimator.predict(X_test)

            train_r2 = r2_score(y_train, y_pred_train)
            train_mse = mean_squared_error(y_train, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_r2 = r2_score(y_test, y_pred_test)
            test_mse = mean_squared_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            results.append({
                'target': target,
                'model_name': model_name,
                'train_r2': train_r2,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'test_r2': test_r2,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'best_estimator':  opt.best_params_,
                'y_test': y_test,
                'y_pred': y_pred_test,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(result_file_path, index=False)


if __name__ == "__main__":
    train_models()