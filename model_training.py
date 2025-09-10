"""
model_training.py

Usage:
  python model_training.py --data data/garments_worker_productivity.csv --target actual_productivity
"""

import argparse
import os
import pandas as pd
import numpy as np
import pickle
from pprint import pprint

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_dataframe(df):
    df = df.copy()
    # If there is a date column, convert to datetime and extract month
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['month'] = df['date'].dt.month
            df.drop(columns=['date'], inplace=True)
            print("Converted 'date' -> 'month'.")
        except Exception as e:
            print("Warning: couldn't convert 'date' column:", e)

    # Strip whitespace & lowercase for string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor, numeric_cols, categorical_cols


def train_and_select(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            random_state=42,
            objective='reg:squarederror',
            verbosity=0
        )
    }

    results = {}
    best_r2 = -np.inf
    best_name, best_pipeline = None, None

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)

        results[name] = {'r2': float(r2), 'mae': float(mae), 'mse': float(mse)}
        print(f"{name} -> R2: {r2:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}")

        if r2 > best_r2:
            best_r2, best_name, best_pipeline = r2, name, pipeline

    return best_name, best_r2, best_pipeline, results


def main(args):
    print("Loading dataset:", args.data)
    df = pd.read_csv(args.data)
    print("Initial shape:", df.shape)
    df = prepare_dataframe(df)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset columns: {df.columns.tolist()}")

    # Drop columns with too many missing values (>60%)
    thresh = 0.6 * len(df)
    df = df.dropna(thresh=thresh, axis=1)
    print("After dropping high-null columns shape:", df.shape)

    # Split X / y
    X = df.drop(columns=[args.target])
    y = df[args.target].values

    # Build preprocessor
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Train models & select best
    best_name, best_r2, best_pipeline, results = train_and_select(X_train, X_test, y_train, y_test, preprocessor)
    print("\nBest model:", best_name, "R2:", best_r2)

    # Save best pipeline
    os.makedirs('models', exist_ok=True)
    out_path = os.path.join('models', args.out)
    metadata = {
        'feature_names': X.columns.tolist(),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'target': args.target,
        'training_results': results,
        'best_model_name': best_name
    }
    with open(out_path, 'wb') as f:
        pickle.dump({'pipeline': best_pipeline, 'meta': metadata}, f)

    print(f"✅ Saved best pipeline to {out_path}")
    print("Training results summary:")
    pprint(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to dataset csv')
    parser.add_argument('--target', required=True, help='name of target column in csv')
    parser.add_argument('--out', default='gwp.pkl', help='output filename to save model in models/')
    parser.add_argument('--test_size', type=float, default=0.2, help='test size fraction')
    args = parser.parse_args()
    main(args)
