"""
Usage: 
    python supermarket_sales_model_builder.py \
        -trp ../data/supermarket-sales/train.csv \
        -tep ../data/supermarket-sales/test.csv \
        -m lin_reg
"""
import argparse
from typing import Tuple
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow

import base_model_builder as bmb

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("supermarket-sales-exp")

def prepare_data(df:pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    target_col = "Item_Outlet_Sales"

    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    return X, y

def log_plot(y_true, y_pred, msg:str):
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(y_true, color="blue", alpha=0.5, label="Actual")
    sns.histplot(y_pred, color="red", alpha=0.5, label="Prediction")
    plt.legend()
    mlflow.log_figure(fig, artifact_file=f"plots/pred_compare_{msg}.png")
    fig.clear(True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", "-trp", required=True,
                    help="Training data file path")
    ap.add_argument("--test-path", "-tep", required=True,
                    help="Testing data file path")
    ap.add_argument("--model", "-m", default="lin_reg",
                    choices=["lin_reg"], help="Model name")
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

        if args.model == "lin_reg":
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_train_pred = lr.predict(X_train)
            y_val_pred = lr.predict(X_val)
            y_test_pred = lr.predict(X_test)

            rmse_calc = lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p))
            mlflow.log_metric("train.rmse", rmse_calc(y_train, y_train_pred))
            mlflow.log_metric("val.rmse", rmse_calc(y_val, y_val_pred))
            mlflow.log_metric("test.rmse", rmse_calc(y_test, y_test_pred))
            
            log_plot(y_train, y_train_pred, msg="train")
            log_plot(y_val, y_val_pred, msg="val")
            log_plot(y_test, y_test_pred, msg="test")
            
            mlflow.sklearn.log_model(lr, "models_mlflow")
