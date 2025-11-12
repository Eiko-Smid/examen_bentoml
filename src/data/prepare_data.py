import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pathlib import Path


DATA_PATH = "data/raw/admission.csv"

COLUMN_NAMES = {
    "LOR ": "LOR"
}

DTYPE_MAPPINGS = {
    "GRE Score":"int32",
    "TOEFL Score": "int32",
    "University Rating": "int64",
    "SOP": "float64",
    "LOR": "float64",
    "CGPA": "float64",
    "Research": "int32",
    "Chance of Admit":"float64"
}

TARGET_VAR = "Chance of Admit"
TEST_SIZE = 0.2
RAND_STATE = 42

PREP_DATA_PATH = "data/processed"
X_TRAIN_FILE = "X_train.csv"
X_TEST_FILE = "X_test.csv"
Y_TRAIN_FILE = "y_train.csv"
Y_TEST_FILE = "y_test.csv"



def load_data(data_path: str):
    df = pd.read_csv(filepath_or_buffer=data_path)
    return df


def preprocess_data(df: pd.DataFrame):
    print(f"\nData view:\n{df.head()}")
    # print(f"\n\nData info:\n{df.info()}")"nans = df.isna().sum(axis="columns")"

    # Drop Serial No
    df = df.drop(columns=["Serial No."])

    # Force column names
    df = df.rename(columns=COLUMN_NAMES)

    # Convert dtypes
    df = df.astype(dtype=DTYPE_MAPPINGS)
    print(f"\nData view:\n{df.head()}")

    # Drop NANs
    nans = df.isna().sum(axis="rows")
    print(f"\nNANs:\n{nans}")
    df_clean = df.dropna(axis="columns")
    
    return df_clean


def split_data(df: pd.DataFrame, test_size: float, rand_state: int):
    X = df.drop(columns=[TARGET_VAR])
    y = df[TARGET_VAR]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=rand_state)
    return X_train, X_test, y_train, y_test


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def store_prep_data(
        data_path: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
):
    data_path_ = Path(data_path)
    data_path_.mkdir(exist_ok=True, parents=True)

    data = [X_train, X_test, y_train, y_test]    
    filenames = [X_TRAIN_FILE, X_TEST_FILE, Y_TRAIN_FILE, Y_TEST_FILE]
    for df, fn in zip(data, filenames):
        path = data_path_ / fn
        df.to_csv(
            path_or_buf=path,
            index=False
        )
    

def preprocessing():
    df = load_data(data_path=DATA_PATH)
    df_clean = preprocess_data(df=df)
    X_train, X_test, y_train, y_test = split_data(
        df=df_clean,
        test_size=TEST_SIZE,
        rand_state=RAND_STATE
    )
    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train=X_train, 
        X_test=X_test
    )
    store_prep_data(
        data_path=PREP_DATA_PATH,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test
    )



if __name__ == "__main__":
    preprocessing()