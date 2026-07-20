"""
Data Preprocessing Module
Handles data cleaning, aggregation, normalization, and feature extraction
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing tasks for energy consumption data"""

    def __init__(self):
        self.data = None
        self.original_data = None
        # default dataset configuration; will be overridden on import
        self.config = {
            "datetime_column": "datetime",
            "target_column": "Global_active_power",
            "feature_columns": [
                "hour",
                "day",
                "month",
                "day_of_week",
                "is_weekend",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
                "Global_reactive_power",
                "Global_intensity",
            ],
            "entity_column": None,
            "frequency": "H",  # "H" for hourly, "D" for daily
        }

    def set_config(self, config: dict):
        """Update dataset configuration safely"""
        if not isinstance(config, dict):
            return
        self.config.update({k: v for k, v in config.items() if v is not None})

    def load_sample_data(self):
        """Generate sample household energy consumption data - DISABLED
        Use CSV import instead to load real data."""
        # Sample data generation is disabled
        # Users should import their own CSV data via /api/data/import
        logger.info("Sample data generation is disabled. Please import your CSV data.")
        return None

    def clean_data(self):
        """
        Clean the data by handling missing values, negative values, and smoothing noise
        This implements Step 4 of the flow: "System internally: Missing readings fill karta hai, Negative values hataata hai, Noise smooth karta hai"
        """
        if self.data is None:
            return None

        # Step 4a: Remove negative values (electricity consumption cannot be negative)
        target_col = self.config.get("target_column", "Global_active_power")
        if target_col in self.data.columns:
            negative_count = (self.data[target_col] < 0).sum()
            if negative_count > 0:
                logger.info("Removing %s negative power values", negative_count)
                self.data = self.data[self.data[target_col] >= 0]

        # Step 4b: Handle missing datetime values
        self.data = self.data.dropna(subset=["datetime"])
        
        # Step 4c: Fill missing numeric values using forward/backward fill, then median
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = (
            self.data[numeric_cols]
            .ffill()
            .bfill()
        )
        
        # Step 4d: Replace infinite values
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        
        # Step 4e: Fill remaining NaN with median
        self.data[numeric_cols] = self.data[numeric_cols].fillna(
            self.data[numeric_cols].median()
        )

        return self.data

    def smooth_noise(self, window_size=5):
        """
        Smooth noise using rolling average - Step 4 of the flow
        "Noise smooth karta hai"
        """
        if self.data is None:
            return None
        
        target_col = self.config.get("target_column", "Global_active_power")
        if target_col not in self.data.columns:
            return self.data
        
        # Apply rolling mean for smoothing
        self.data[target_col] = self.data[target_col].rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
        
        return self.data

    def aggregate_by_day(self):
        """Aggregate data to daily level"""
        if self.data is None:
            return None
        
        dt_col = self.config.get("datetime_column", "datetime")
        df = self.data.copy()
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.set_index(dt_col)
        
        # Select only numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]
        
        daily_data = df_numeric.resample("D").mean()
        
        # Reset index to get datetime back as column
        daily_data = daily_data.reset_index()
        return daily_data

    def aggregate_by_month(self):
        """Aggregate data to monthly level"""
        if self.data is None:
            return None
        
        dt_col = self.config.get("datetime_column", "datetime")
        df = self.data.copy()
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col])
            df = df.set_index(dt_col)
        
        # Select only numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]
        
        monthly_data = df_numeric.resample("M").mean()
        
        # Reset index to get datetime back as column
        monthly_data = monthly_data.reset_index()
        return monthly_data

    def extract_features(self):
        """Extract time-based features from datetime"""
        if self.data is None:
            return self.data
        dt_col = self.config.get("datetime_column", "datetime")
        frequency = (self.config.get("frequency") or "H").upper()
        if dt_col not in self.data.columns:
            return self.data

        # Make a copy to avoid modifying the original data
        self.data = self.data.copy()

        # Ensure datetime column is properly parsed
        if not pd.api.types.is_datetime64_any_dtype(self.data[dt_col]):
            try:
                self.data[dt_col] = pd.to_datetime(
                    self.data[dt_col], errors="coerce"
                )
            except Exception as e:
                logger.warning("Could not parse datetime column: %s", e)
                return self.data

        # Check if datetime parsing was successful
        if not pd.api.types.is_datetime64_any_dtype(self.data[dt_col]):
            return self.data

        dt_series = self.data[dt_col]

        # Extract time features conditionally based on frequency
        if "day" not in self.data.columns:
            self.data["day"] = dt_series.dt.day
        if "month" not in self.data.columns:
            self.data["month"] = dt_series.dt.month
        if "year" not in self.data.columns:
            self.data["year"] = dt_series.dt.year
        if "day_of_week" not in self.data.columns:
            self.data["day_of_week"] = dt_series.dt.dayofweek
        if "is_weekend" not in self.data.columns:
            self.data["is_weekend"] = (
                self.data["day_of_week"].isin([5, 6]).astype(int)
            )

        # Hourly-specific features
        if frequency == "H":
            if "hour" not in self.data.columns:
                self.data["hour"] = dt_series.dt.hour
            if "hour_sin" not in self.data.columns:
                self.data["hour_sin"] = np.sin(
                    2 * np.pi * self.data["hour"] / 24
                )
            if "hour_cos" not in self.data.columns:
                self.data["hour_cos"] = np.cos(
                    2 * np.pi * self.data["hour"] / 24
                )
        else:
            # Ensure hourly-only columns are removed to avoid confusion
            for col in ["hour", "hour_sin", "hour_cos"]:
                if col in self.data.columns:
                    self.data.drop(columns=[col], inplace=True, errors="ignore")

        # Month cyclic encoding (works for both frequencies)
        if "month_sin" not in self.data.columns:
            self.data["month_sin"] = np.sin(
                2 * np.pi * self.data["month"] / 12
            )
        if "month_cos" not in self.data.columns:
            self.data["month_cos"] = np.cos(
                2 * np.pi * self.data["month"] / 12
            )

        return self.data

    def get_consumption_summary(self):
        """Get summary statistics of energy consumption"""
        if self.data is None:
            return None
        target_col = self.config.get("target_column", "Global_active_power")
        entity_col = self.config.get("entity_column")
        dt_col = self.config.get("datetime_column", "datetime")

        def safe_num(val: float) -> float:
            try:
                import math
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    return 0.0
                return float(val)
            except Exception:
                return 0.0

        def col_stats(series):
            if series is None or series.empty:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
            return {
                "mean": safe_num(series.mean()),
                "std": safe_num(series.std()),
                "min": safe_num(series.min()),
                "max": safe_num(series.max()),
                "total": safe_num(series.sum()),
            }

        summary = {
            "total_records": len(self.data),
            "date_range": {
                "start": (
                    str(self.data.index.min())
                    if isinstance(self.data.index, pd.DatetimeIndex)
                    else str(self.data[dt_col].min())
                    if dt_col in self.data.columns
                    else None
                ),
                "end": (
                    str(self.data.index.max())
                    if isinstance(self.data.index, pd.DatetimeIndex)
                    else str(self.data[dt_col].max())
                    if dt_col in self.data.columns
                    else None
                ),
            },
            "target": {
                "name": target_col,
                **col_stats(self.data.get(target_col)),
            },
        }

        # Preserve legacy keys if columns exist for backward compatibility
        if "Global_active_power" in self.data.columns:
            summary["global_active_power"] = {
                **col_stats(self.data.get("Global_active_power"))
            }
        if "Global_intensity" in self.data.columns:
            summary["global_intensity"] = {
                **col_stats(self.data.get("Global_intensity"))
            }
        if "Voltage" in self.data.columns:
            summary["voltage"] = {**col_stats(self.data.get("Voltage"))}
        if (
            "Sub_metering_1" in self.data.columns
            or "Sub_metering_2" in self.data.columns
            or "Sub_metering_3" in self.data.columns
        ):
            summary["sub_metering"] = {
                "1_mean": float(self.data.get("Sub_metering_1", 0).mean())
                if isinstance(self.data.get("Sub_metering_1"), pd.Series)
                else 0,
                "2_mean": float(self.data.get("Sub_metering_2", 0).mean())
                if isinstance(self.data.get("Sub_metering_2"), pd.Series)
                else 0,
                "3_mean": float(self.data.get("Sub_metering_3", 0).mean())
                if isinstance(self.data.get("Sub_metering_3"), pd.Series)
                else 0,
            }
        if entity_col and entity_col in self.data.columns:
            summary["entities"] = int(self.data[entity_col].nunique())

        return summary

    def get_hourly_pattern(self):
        """Get average consumption by hour of day"""
        target_col = self.config.get("target_column", "Global_active_power")
        if (
            self.data is None
            or "hour" not in self.data.columns
            or target_col not in self.data.columns
        ):
            return []
        hourly = (
            self.data.groupby("hour")[target_col]
            .mean()
            .reset_index()
        )
        return hourly.to_dict("records")

    def get_daily_pattern(self):
        """Get average consumption by day of week"""
        target_col = self.config.get("target_column", "Global_active_power")
        if (
            self.data is None
            or "day_of_week" not in self.data.columns
            or target_col not in self.data.columns
        ):
            return []
        daily = (
            self.data.groupby("day_of_week")[target_col]
            .mean()
            .reset_index()
        )
        return daily.to_dict("records")

    def get_monthly_pattern(self):
        """Get average consumption by month"""
        target_col = self.config.get("target_column", "Global_active_power")
        if (
            self.data is None
            or "month" not in self.data.columns
            or target_col not in self.data.columns
        ):
            return []
        monthly = (
            self.data.groupby("month")[target_col]
            .mean()
            .reset_index()
        )
        return monthly.to_dict("records")

    def get_peak_hours(self):
        """Identify peak usage hours"""
        target_col = self.config.get("target_column", "Global_active_power")
        if (
            self.data is None
            or "hour" not in self.data.columns
            or target_col not in self.data.columns
        ):
            return {"peak_hours": [], "threshold": 0.0, "average": 0.0}
        hourly_avg = self.data.groupby("hour")[target_col].mean()
        mean_val = float(hourly_avg.mean()) if not hourly_avg.empty else 0.0
        std_val = float(hourly_avg.std()) if not hourly_avg.empty else 0.0
        if np.isnan(std_val):
            std_val = 0.0
        threshold = mean_val + std_val
        peak_hours = (
            hourly_avg[hourly_avg > threshold].index.tolist()
            if threshold is not None
            else []
        )
        return {
            "peak_hours": peak_hours,
            "threshold": float(threshold),
            "average": mean_val,
        }

    def get_recent_daily_summary(self, days: int = 3):
        """Return per-day stats for the most recent N days (defaults to last 3)."""
        if self.data is None or days <= 0:
            return []

        dt_col = self.config.get("datetime_column", "datetime")
        target_col = self.config.get("target_column", "Global_active_power")

        if dt_col not in self.data.columns or target_col not in self.data.columns:
            return []

        df = self.data.copy()
        # Parse datetime column safely
        if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col])
        if df.empty:
            return []

        # Compute date range based on most recent timestamp
        latest_ts = df[dt_col].max()
        start_ts = latest_ts.normalize() - pd.Timedelta(days=days - 1)
        recent = df[df[dt_col] >= start_ts]
        if recent.empty:
            return []

        recent["date"] = recent[dt_col].dt.date
        grouped = recent.groupby("date")[target_col].agg(["mean", "min", "max", "sum", "count"]).reset_index()
        grouped = grouped.sort_values("date", ascending=True)

        return [
            {
                "date": str(row["date"]),
                "mean": float(row["mean"]) if pd.notna(row["mean"]) else 0.0,
                "min": float(row["min"]) if pd.notna(row["min"]) else 0.0,
                "max": float(row["max"]) if pd.notna(row["max"]) else 0.0,
                "total": float(row["sum"]) if pd.notna(row["sum"]) else 0.0,
                "count": int(row["count"]) if pd.notna(row["count"]) else 0,
            }
            for _, row in grouped.iterrows()
        ]


preprocessor = DataPreprocessor()
