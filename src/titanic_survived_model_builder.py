"""
Usage: 
    python titanic_survived_model_builder.py \
        -trp ../data/titanic/train-data.csv \
        -tep ../data/titanic/test-data.csv \
        -m log_reg
"""
import argparse
from typing import Tuple
import mlflow

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import base_model_builder as bmb

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic-survived-exp")


def prepare_data(df:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    target_col = "Survived"

    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    return X, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", "-trp", required=True,
                    help="Training data file path")
    ap.add_argument("--test-path", "-tep", required=True,
                    help="Testing data file path")
    ap.add_argument("--model", "-m", default="log_reg",
                    choices=["log_reg"], help="Model name")
    args = ap.parse_args()

    with mlflow.start_run():
        mlflow.set_tag("developer", "peeush-agarwal")
        mlflow.set_tag("model", args.model)
        mlflow.log_param("training path", args.train_path)
        mlflow.log_param("testing path", args.test_path)

        df = bmb.load_data(args.train_path)
        df_test = bmb.load_data(args.test_path)

        df_train, df_val = bmb.split_data(df)

        X_train, y_train = prepare_data(df_train)
        X_val, y_val = prepare_data(df_val)
        X_test, y_test = prepare_data(df_test)

        if args.model == "log_reg":
            lr = LogisticRegression()
            lr.fit(X_train, y_train)

            y_train_pred = lr.predict(X_train)
            y_val_pred = lr.predict(X_val)
            y_test_pred = lr.predict(X_test)

            acc_calc = lambda y_t, y_p: accuracy_score(y_t, y_p)
            mlflow.log_metric("train_acc", acc_calc(y_train, y_train_pred))
            mlflow.log_metric("val_acc", acc_calc(y_val, y_val_pred))
            mlflow.log_metric("test_acc", acc_calc(y_test, y_test_pred))

            mlflow.sklearn.log_model(lr, "models_mlflow")
