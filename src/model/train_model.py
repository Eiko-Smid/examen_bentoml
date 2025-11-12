import os
from os import PathLike

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

import bentoml

RAND_STATE = 42

PREP_DATA_PATH = "data/processed"
X_TRAIN_FILE = "X_train.csv"
X_TEST_FILE = "X_test.csv"
Y_TRAIN_FILE = "y_train.csv"
Y_TEST_FILE = "y_test.csv"

PARAM_GRID= {
        "model__n_estimators": [100, 300, 500],
        "model__learning_rate": [0.1, 0.05, 0.02],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 1.0],
    }
SCORING = "r2"
SCORE_THRESHOLD = 0.78

MODEL_NAME = "student_admissions_predictor"
TARGET_VAR = "Chance of Admit"



def load_data_split(data_path: str | PathLike[str]):
    dfs = []
    filenames = [X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE]

    data_path_ = Path(data_path)
    if data_path_.exists():
        for fn in filenames:
            path = data_path_ / fn
            dfs.append(pd.read_csv(filepath_or_buffer=path))

    return dfs


def build_pipeline(random_state: int) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("model", GradientBoostingRegressor(random_state=random_state)),
        ]
    )
    return pipe


def train_model(X_train, X_test, y_train, y_test):
    # Make y flat
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]
    
    # Define pipeline
    pipe = build_pipeline(random_state=RAND_STATE)

    # Define training
    cv = KFold(n_splits=5, shuffle=True, random_state=RAND_STATE)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRID,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    # Train model
    print("Running GridSearchCV...")
    grid.fit(
        X=X_train,
        y=y_train
    )

    # Print info
    best_params = grid.best_params_
    best_cv_score = grid.best_score_
    best_estimator = grid.best_estimator_
    print(f"Best params: {best_params}")
    print(f"Best CV R²: {best_cv_score:.4f}")

    # Evaluate on test set
    y_pred = best_estimator.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Test R²:   {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    # Retrain on whole dataset and store model in bento
    if test_r2 > SCORE_THRESHOLD:
        # Build whole test set
        X = pd.concat([X_train, X_test], axis=0)
        y = pd.concat([y_train, y_test], axis=0)

        final_pipe = build_pipeline(RAND_STATE)
        final_pipe.set_params(**best_params)
        final_pipe.fit(X, y)

        # Save to BentoML
        print("Saving model to BentoML...")
        model_ref = bentoml.sklearn.save_model(
            name=MODEL_NAME,
            model=final_pipe,
            signatures={
                # This declares the method you'll call in your service:
                # svc = bentoml.sklearn.get("name:tag").to_runner(); runner.predict.run(df)
                "predict": {"batchable": True}
            },
            metadata={
                "cv_mean_r2": best_cv_score,
                "test_r2": float(test_r2),
                "test_rmse": float(test_rmse),
                "best_params": best_params,
                "features": list(X.columns),
                "target": TARGET_VAR,
            },
        )

        print(f"\nSaved model as: {model_ref}")

    
def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data_split(data_path=PREP_DATA_PATH)
    # data = load_data_split(data_path=PREP_DATA_PATH)
    # for df in data:
    #     print(f"\n\nData:\n{df.head()}")

    # Train model
    train_model(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    


if __name__ == "__main__":
    main()