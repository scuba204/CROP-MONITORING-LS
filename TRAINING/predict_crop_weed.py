#!/usr/bin/env python3
"""
predict_crop_weed.py
Updated to handle large datasets with Drive export and safe Earth Engine initialization.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import joblib
import ee

# Ensure project root is on path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.extract_features import extract_spectral_features, initialize_ee_project

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Predict crop vs. weed using Sentinel-2 features.")
    parser.add_argument('--input-csv', type=str, default='unlabelled_field_data.csv')
    parser.add_argument('--model-name', type=str, default='crop_vs_weed_model.pkl')
    parser.add_argument('--features-file', type=str, default='feature_list.json')
    parser.add_argument('--output-csv', type=str, default='predictions_output.csv')
    parser.add_argument('--start-date', type=str, default='2025-07-01')
    parser.add_argument('--end-date', type=str, default='2025-08-05')
    parser.add_argument('--drive-folder', type=str, default=None, help='Google Drive folder for large exports')
    return parser.parse_args()

# --- Vegetation Index Calculations ---
def calculate_vegetation_indices(df: pd.DataFrame) -> pd.DataFrame:
    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'])
    df['EVI'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 6 * df['B4'] - 7.5 * df['B3'] + 1)
    df['NDWI'] = (df['B8'] - df['B11']) / (df['B8'] + df['B11'])
    df['NDRE'] = (df['B8'] - df['B5']) / (df['B8'] + df['B5'])
    df['GNDVI'] = (df['B8'] - df['B3']) / (df['B8'] + df['B3'])
    df['MSI'] = df['B11'] / df['B8']
    df['OSAVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 0.16) * (1 + 0.16)
    df['SAVI'] = ((df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 0.5)) * 1.5
    df['RVI'] = df['B8'] / df['B4']
    df.replace([float('inf'), float('-inf')], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

# --- Delta Feature Calculations ---
def calculate_delta_features(df: pd.DataFrame, features_from_model: list) -> pd.DataFrame:
    unique_dates = sorted(df['date'].unique())
    if len(unique_dates) < 2:
        latest_date_str = unique_dates[0].strftime('%Y%m%d')
        data_dict = {}
        for feature_name in features_from_model:
            if latest_date_str in feature_name and '_Δ_' not in feature_name:
                base_metric = feature_name.split('_')[0]
                data_dict[feature_name] = df.loc[df['date'] == unique_dates[0], base_metric].values
            else:
                data_dict[feature_name] = np.zeros(len(df[df['date'] == unique_dates[0]]))
        df_pivoted = pd.DataFrame(data_dict)
        df_pivoted['id'] = df[df['date'] == unique_dates[0]]['id'].values
        df_pivoted.set_index('id', inplace=True)
    else:
        df_pivoted = df.pivot_table(index='id', columns='date', values=['NDVI','EVI','NDWI','NDRE','GNDVI','MSI','OSAVI','SAVI','RVI'])
        df_pivoted.columns = [f'{col[0]}_{col[1].strftime("%Y%m%d")}' for col in df_pivoted.columns]
        first_date_str = unique_dates[0].strftime('%Y%m%d')
        last_date_str = unique_dates[-1].strftime('%Y%m%d')
        delta_metrics = ['NDVI','EVI','NDWI','NDRE','GNDVI','MSI','OSAVI','SAVI','RVI']
        for metric in delta_metrics:
            df_pivoted[f'{metric}_Δ_{last_date_str}'] = df_pivoted[f'{metric}_{last_date_str}'] - df_pivoted[f'{metric}_{first_date_str}']
    return df_pivoted

# --- Main ---
def main():
    args = parse_args()

    model_dir = 'models'
    data_dir = 'data'
    spectral_bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']

    # Load model and feature list
    model_path = os.path.join(model_dir, args.model_name)
    features_path = os.path.join(model_dir, args.features_file)
    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        features = json.load(f)

    # Load unlabeled data
    csv_path = os.path.join(data_dir, args.input_csv)
    df_raw = pd.read_csv(csv_path)
    df_raw['id'] = range(len(df_raw))
    gdf = gpd.GeoDataFrame(df_raw, geometry=[Point(xy) for xy in zip(df_raw.longitude, df_raw.latitude)], crs='EPSG:4326')

    # Extract features
    df_features_or_task = extract_spectral_features(
        gdf, args.start_date, args.end_date, spectral_bands,
        export_to_drive=True, drive_folder=args.drive_folder
    )

    if isinstance(df_features_or_task, ee.batch.Task):
        print(f"Drive export started. Task ID: {df_features_or_task.id}")
        return

    if df_features_or_task.empty:
        print("❌ No features extracted. Exiting.")
        return

    df_features = df_features_or_task

    # Feature engineering
    df_features = calculate_vegetation_indices(df_features)
    df_features = calculate_delta_features(df_features, features)

    # Align features and predict
    X_to_predict = df_features[features]
    predictions = model.predict(X_to_predict)
    df_features['predicted_label'] = predictions

    # Save output
    os.makedirs(data_dir, exist_ok=True)
    output_df = df_raw[['longitude','latitude','id']].merge(df_features[['predicted_label']], on='id')
    output_path = os.path.join(data_dir, 'predictions_output.csv')
    output_df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

if __name__ == '__main__':
    main()
