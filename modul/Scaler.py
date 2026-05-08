import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataFrameScaler:
    def __init__(self, eps=1e-6):
        self.scaler = StandardScaler()
        self.cols = None
        self.eps = eps

    def fit(self, df: pd.DataFrame, cols: list[str]):
        self.cols = cols

        X = df[cols].values.astype(np.float64)

        std = X.std(axis=0)
        bad = std < self.eps
        if bad.any():
            X[:, bad] = 0.0
            print(f"[Scaler] Constant cols fixed: {[cols[i] for i in np.where(bad)[0]]}")

        self.scaler.fit(X)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.cols].values.astype(np.float64)
        Xn = self.scaler.transform(X)
        df.loc[:, self.cols] = Xn
        return df

    def inverse_transform(self, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        if cols is None:
            cols = self.cols

        idx = [self.cols.index(c) for c in cols]

        Xn = df[self.cols].values.astype(np.float64)
        X = self.scaler.inverse_transform(Xn)

        df.loc[:, cols] = X[:, idx]
        return df


class DataFrameScalerAuto:
    def __init__(self, eps=1e-6):
        self.scaler = StandardScaler()
        self.cols = None
        self.eps = eps
        self.binary_cols = []
        self.continuous_cols = []

    def _is_binary_column(self, df: pd.DataFrame, col: str) -> bool:
        if col not in df.columns:
            return False

        unique_vals = df[col].dropna().unique()

        if len(unique_vals) <= 2:
            is_binary = all(v in {0, 1, 0.0, 1.0} for v in unique_vals)
            return is_binary
        return False

    def fit(self, df: pd.DataFrame, cols: list[str]):
        self.cols = cols

        self.binary_cols = []
        self.continuous_cols = []

        for col in cols:
            if self._is_binary_column(df, col):
                self.binary_cols.append(col)
            else:
                self.continuous_cols.append(col)

        print(f"[Scaler] Detected {len(self.binary_cols)} binary columns (no normalization): {self.binary_cols}")
        print(f"[Scaler] Normalizing {len(self.continuous_cols)} continuous columns")

        if self.continuous_cols:
            X = df[self.continuous_cols].values.astype(np.float64)

            std = X.std(axis=0)
            bad = std < self.eps
            if bad.any():
                X[:, bad] = 0.0
                print(f"[Scaler] Constant cols fixed: {[self.continuous_cols[i] for i in np.where(bad)[0]]}")

            self.scaler.fit(X)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()

        if self.continuous_cols:
            X = df_copy[self.continuous_cols].values.astype(np.float64)
            Xn = self.scaler.transform(X)
            df_copy.loc[:, self.continuous_cols] = Xn

        for col in self.binary_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].round().astype(np.int64)

        return df_copy

    def inverse_transform(self, df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        df_copy = df.copy()

        if cols is None:
            inverse_cols = self.continuous_cols
        else:
            inverse_cols = [c for c in cols if c in self.continuous_cols]

        if inverse_cols:
            idx = [self.continuous_cols.index(c) for c in inverse_cols]
            Xn = df_copy[self.continuous_cols].values.astype(np.float64)
            X = self.scaler.inverse_transform(Xn)
            df_copy.loc[:, inverse_cols] = X[:, idx]

        return df_copy

    def get_feature_types(self) -> dict:
        return {
            'binary': self.binary_cols,
            'continuous': self.continuous_cols,
            'total': self.cols
        }