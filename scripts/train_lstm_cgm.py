#!/usr/bin/env python3
"""
Script: train_lstm_cgm.py
Location: cgm/scripts/

This script reads CGM data from cgm/data/, preprocesses it into sequences,
trains an LSTM model to predict the next glucose value or similar models,
and saves the best model and scaler.

Directory notation:
  - Use a trailing slash to denote a folder (e.g., "cgm/data/").
  - A leading slash ("/path/to/folder") denotes an absolute path from the root.

Usage:
  cd cgm
  python3 -m venv venv            # create virtual environment
  source venv/bin/activate        # macOS/Linux (or venv\\Scripts\\activate on Windows)
  pip install -r requirements.txt # requirements.txt should include pandas, numpy,
                                  # scikit-learn, tensorflow, joblib

  cd scripts
  python train_lstm_cgm.py \
      --data-file ../data/Clarity_Export_Harlow_Iain_2025-05-20_203133.csv \
      --seq-length 24 --epochs 50 --batch-size 32 --model-dir ../models

For interactive exploration, open a Jupyter notebook in cgm/notebooks/ and
import functions from this script.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib


def load_data(file_path):
    """
    Load CGM CSV, automatically detect timestamp and glucose columns,
    parse dates, set index, and return a DataFrame with a single
    'Glucose (mg/dL)' column.
    """
    df = pd.read_csv(file_path)
    # Detect timestamp column
    ts_col = next((c for c in df.columns if 'Timestamp' in c), None)
    if ts_col is None:
        raise ValueError('No timestamp column found in CSV')
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col)
    # Detect glucose column
    glucose_col = next((c for c in df.columns if 'Glucose' in c), None)
    if glucose_col is None:
        raise ValueError('No glucose column found in CSV')
    df = df[[glucose_col]].dropna()
    df.columns = ['Glucose (mg/dL)']
    return df


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def main(args):
    data_file = args.data_file
    seq_length = args.seq_length
    model_dir = args.model_dir

    os.makedirs(model_dir, exist_ok=True)
    df = load_data(data_file)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X, y = create_sequences(scaled, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_model((seq_length, 1))
    checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'best_model.h5'),
        monitor='val_loss', save_best_only=True, verbose=1
    )
    history = model.fit(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=[checkpoint]
    )

    joblib.dump(scaler, os.path.join(model_dir, 'scaler.save'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an LSTM on CGM data from Clarity export'
    )
    parser.add_argument(
        '--data-file', type=str,
        default='../data/Clarity_Export_Harlow_Iain_2025-05-20_203133.csv',
        help='Path to CGM CSV file'
    )
    parser.add_argument(
        '--seq-length', type=int, default=24,
        help='Number of timesteps per training sequence'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--model-dir', type=str, default='../models',
        help='Directory to save model and scaler'
    )
    args = parser.parse_args()
    main(args)
