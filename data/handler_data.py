import os
import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def handle_missing_values_optimized(df, verbose=True):
    df_clean = df.copy()

    if verbose:
        print(f"\n  Original shape: {df.shape}")
        print(f"  Original missing values: {df.isnull().sum().sum()}")
        print(f"  Original infinite values: {df.isin([np.inf, -np.inf]).sum().sum()}")

    ttc_cols = ['ttc_preceding', 'ttc_following']
    for col in ttc_cols:
        if col in df_clean.columns:
            inf_mask = df_clean[col].isin([np.inf, -np.inf])
            if inf_mask.any():
                df_clean.loc[inf_mask, col] = 10.0
                if verbose:
                    print(f"    {col}: {inf_mask.sum()} infinite values -> 10.0")

    thw_cols = ['thw_preceding', 'thw_following']
    for col in thw_cols:
        if col in df_clean.columns:
            inf_mask = df_clean[col].isin([np.inf, -np.inf])
            if inf_mask.any():
                df_clean.loc[inf_mask, col] = 5.0
                if verbose:
                    print(f"    {col}: {inf_mask.sum()} infinite values -> 5.0")

    gap_cols = ['preceding_gap', 'following_gap', 'left_gap', 'right_gap', 'left_lat_gap', 'right_lat_gap']
    for col in gap_cols:
        if col in df_clean.columns:
            inf_mask = df_clean[col].isin([np.inf, -np.inf])
            if inf_mask.any():
                df_clean.loc[inf_mask, col] = 0
                if verbose:
                    print(f"    {col}: {inf_mask.sum()} infinite values -> 0")

    no_car_cols = [
        'preceding_v', 'preceding_delta_v',
        'following_v', 'following_delta_v',
        'left_v', 'left_delta_v',
        'right_v', 'right_delta_v'
    ]
    for col in no_car_cols:
        if col in df_clean.columns:
            missing_mask = df_clean[col].isnull()
            if missing_mask.any():
                df_clean.loc[missing_mask, col] = 0
                if verbose:
                    print(f"    {col}: {missing_mask.sum()} missing values -> 0")

    lc_cols = ['accepted_gap_ratio']
    for col in lc_cols:
        if col in df_clean.columns:
            missing_mask = df_clean[col].isnull()
            if missing_mask.any():
                df_clean.loc[missing_mask, col] = 0
                if verbose:
                    print(f"    {col}: {missing_mask.sum()} missing values -> 0")

    ego_cont_cols = ['ego_ax', 'ego_ay', 'ego_vx', 'ego_vy', 'ego_x', 'ego_y', 'ego_delta']
    for col in ego_cont_cols:
        if col in df_clean.columns:
            missing_mask = df_clean[col].isnull()
            if missing_mask.any():
                df_clean[col] = df_clean[col].fillna(method='ffill')
                if df_clean[col].isnull().any():
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                if verbose:
                    print(f"    {col}: {missing_mask.sum()} missing values -> forward fill + mean")

    ego_cat_cols = ['id', 'frame', 'lane_id']
    for col in ego_cat_cols:
        if col in df_clean.columns:
            missing_mask = df_clean[col].isnull()
            if missing_mask.any():
                df_clean[col] = df_clean[col].fillna(method='ffill')
                if df_clean[col].isnull().any():
                    if col == 'lane_id':
                        df_clean[col] = df_clean[col].fillna(4)
                    else:
                        df_clean[col] = df_clean[col].fillna(0)
                if verbose:
                    print(f"    {col}: {missing_mask.sum()} missing values -> forward fill")

    other_cols = [col for col in df_clean.columns if df_clean[col].isnull().any()]
    for col in other_cols:
        if col not in no_car_cols + lc_cols + ego_cont_cols + ego_cat_cols:
            missing_mask = df_clean[col].isnull()
            if missing_mask.any():
                median_val = df_clean[col].median()
                df_clean.loc[missing_mask, col] = median_val
                if verbose:
                    print(f"    {col}: {missing_mask.sum()} missing values -> median {median_val:.4f}")

    final_missing = df_clean.isnull().sum().sum()
    final_inf = df_clean.isin([np.inf, -np.inf]).sum().sum()

    if verbose:
        print(f"\n  Final missing values: {final_missing}")
        print(f"  Final infinite values: {final_inf}")

        if final_missing == 0 and final_inf == 0:
            print("  Data cleaning completed, no missing or infinite values")
        else:
            print("  Warning: Data issues still exist")

    return df_clean


def process_feature_file(input_path, output_path, verbose=True):
    print(f"\n{'='*60}")
    print(f"Processing file: {os.path.basename(input_path)}")
    print(f"{'='*60}")

    try:
        df = pd.read_csv(input_path)
        print(f"  Read successfully, shape: {df.shape}")

        df_clean = handle_missing_values_optimized(df, verbose=verbose)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)

        print(f"\n  Saved to: {output_path}")
        return True

    except Exception as e:
        print(f"  Failed: {e}")
        return False


def batch_process_all_features(input_dir, output_dir, pattern="*_tracks_extract_feature*.csv"):
    input_files = glob.glob(os.path.join(input_dir, pattern))

    if not input_files:
        print(f"No files found: {input_dir}/{pattern}")
        return

    print(f"\n{'='*80}")
    print(f"Batch processing feature files")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(input_files)} files")

    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    for input_path in sorted(input_files):
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        if process_feature_file(input_path, output_path):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"Processing completed: Success {success_count}/{len(input_files)}")
    print(f"{'='*80}")


def quick_analyze(file_path):
    print(f"\n{'='*60}")
    print(f"Analyzing file: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape}")

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\nColumns with missing values ({len(missing_cols)}):")
        for col, cnt in missing_cols.items():
            pct = cnt / len(df) * 100
            print(f"  {col}: {cnt} ({pct:.2f}%)")

    inf_counts = df.isin([np.inf, -np.inf]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    if len(inf_cols) > 0:
        print(f"\nColumns with infinite values ({len(inf_cols)}):")
        for col, cnt in inf_cols.items():
            pct = cnt / len(df) * 100
            print(f"  {col}: {cnt} ({pct:.2f}%)")

    return df


if __name__ == "__main__":
    INPUT_DIR = "allFeature_traj_fixed_features"
    OUTPUT_DIR = "allFeature_traj_fixed_features_clean"
    FILE_PATTERN = "*_tracks_extract_feature.csv"

    sample_files = glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN))
    if sample_files:
        quick_analyze(sample_files[0])
        input("\nPress Enter to continue processing...")

    batch_process_all_features(INPUT_DIR, OUTPUT_DIR, FILE_PATTERN)

    print(f"\nProcessing completed, files saved in: {OUTPUT_DIR}")