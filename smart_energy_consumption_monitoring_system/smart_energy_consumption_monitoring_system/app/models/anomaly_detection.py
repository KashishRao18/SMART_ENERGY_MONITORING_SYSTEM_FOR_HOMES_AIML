"""
Anomaly Detection Module
Implements Isolation Forest for detecting abnormal energy consumption patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List


class AnomalyDetector:
    """Detects abnormal energy consumption patterns using Isolation Forest"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            "Global_active_power",
            "Global_reactive_power",
            "Global_intensity",
            "Voltage",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]
        self.is_trained = False
        self.contamination = 0.01
        self.config = {
            "target_column": "Global_active_power",
            "datetime_column": "datetime",
            "entity_column": None,
        }

    def set_config(self, config: dict):
        """Update configuration for feature selection"""
        if not isinstance(config, dict):
            return
        self.config.update({k: v for k, v in config.items() if v is not None})
        # If feature_columns provided, override defaults
        if "feature_columns" in config and config["feature_columns"]:
            self.feature_columns = config["feature_columns"]

    def prepare_features(self, data):
        """Prepare features for anomaly detection"""
        if data is None:
            return None

        # Make a copy to avoid modifying the original data
        data = data.copy()

        # Dynamically infer feature columns (numeric, excluding target/datetime/entity)
        target_col = self.config.get("target_column", "Global_active_power")
        dt_col = self.config.get("datetime_column", "datetime")
        entity_col = self.config.get("entity_column")

        available_features = [
            col
            for col in data.columns
            if pd.api.types.is_numeric_dtype(data[col])
            and col not in [dt_col, entity_col]
        ]

        # Drop identifier-like numeric columns (e.g., Household_ID) that distort anomalies
        pruned_features = []
        n_rows = max(len(data), 1)
        for col in available_features:
            col_l = str(col).lower()
            unique_ratio = data[col].nunique(dropna=True) / n_rows
            is_id_like = ("id" in col_l and unique_ratio > 0.8)
            if not is_id_like:
                pruned_features.append(col)
        available_features = pruned_features

        # fallback to explicit list if inference failed
        if not available_features:
            available_features = [
                col for col in self.feature_columns if col in data.columns
            ]

        if not available_features:
            return None

        X = data[available_features].copy()
        
        # Handle NaN values - replace with median
        if X.isnull().any().any():
            X = X.fillna(X.median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        if X.isnull().any().any():
            X = X.fillna(0)

        # Ensure X is always 2D (required by sklearn)
        if isinstance(X, pd.DataFrame):
            X_values = X.values

            # Ensure X is 2D
            if X_values.ndim == 1:
                X_values = X_values.reshape(-1, 1)

            return X_values

        return X

    def train(self, data, contamination=0.01):
        """Train the Isolation Forest model"""
        X = self.prepare_features(data)
        if X is None or len(X) == 0:
            return {"error": "Insufficient data for training"}

        # Keep contamination in a stable, practical range.
        contamination = float(contamination)
        contamination = min(max(contamination, 0.001), 0.2)
        self.contamination = contamination
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_trained = True

        return {
            "status": "trained",
            "n_samples": len(X),
            "contamination": contamination,
            "n_features": X.shape[1],
        }

    def detect_anomalies(self, data):
        """Detect anomalies in energy consumption data"""
        if not self.is_trained or self.model is None:
            return {"error": "Model not trained"}

        X = self.prepare_features(data)
        if X is None or len(X) == 0:
            return {"error": "No data to analyze"}

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        results = data.copy()
        results["anomaly"] = predictions
        results["anomaly_score"] = scores
        results["is_anomaly"] = results["anomaly"] == -1

        anomaly_count = (predictions == -1).sum()
        normal_count = (predictions == 1).sum()

        anomalies = results[results["is_anomaly"]].copy()

        anomaly_details = []
        if len(anomalies) > 0:
            for idx, row in anomalies.iterrows():
                detail = {}
                if "datetime" in row:
                    detail["datetime"] = str(row["datetime"])
                if "Global_active_power" in row:
                    detail["power"] = float(row["Global_active_power"])
                detail["score"] = float(row["anomaly_score"])
                anomaly_details.append(detail)

        return {
            "total_records": len(data),
            "anomaly_count": int(anomaly_count),
            "normal_count": int(normal_count),
            "anomaly_percentage": float(anomaly_count / len(data) * 100),
            "anomalies": anomaly_details,
            "data": (
                self._build_serializable_timeseries(results)
                if "datetime" in results.columns
                else None
            ),
        }

    def _build_serializable_timeseries(self, results: pd.DataFrame) -> List[Dict[str, Any]]:
        """Return chart-ready anomaly rows with stable ISO-like datetime strings."""
        frame = results[
            ["datetime", "Global_active_power", "anomaly_score", "is_anomaly"]
        ].copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.dropna(subset=["datetime"])
        frame["datetime"] = frame["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return frame.to_dict("records")

    def get_anomaly_patterns(self, data):
        """Analyze patterns in detected anomalies"""
        if not self.is_trained:
            return {"error": "Model not trained"}

        results = self.detect_anomalies(data)
        if "error" in results:
            return results

        anomalies_raw = results.get("anomalies", [])
        anomalies: List[Dict[str, Any]] = []
        if isinstance(anomalies_raw, list):
            anomalies = [a for a in anomalies_raw if isinstance(a, dict)]

        if not anomalies:
            return {
                "patterns": [],
                "insight": "Energy consumption is within normal range",
            }

        patterns = []

        high_power_anomalies = [
            a for a in anomalies if a.get("power", 0) > 3.0
        ]
        if high_power_anomalies:
            patterns.append(
                {
                    "type": "High Consumption",
                    "count": len(high_power_anomalies),
                    "description": "Detected unusually high power consumption",
                }
            )

        low_power_anomalies = [a for a in anomalies if a.get("power", 0) < 0.2]
        if low_power_anomalies:
            patterns.append(
                {
                    "type": "Very Low Consumption",
                    "count": len(low_power_anomalies),
                    "description": "Detected unusually low power consumption",
                }
            )

        if len(anomalies) < len(data) * 0.02:
            insight = "Energy consumption is mostly normal with few anomalies"
        elif len(anomalies) < len(data) * 0.05:
            insight = "Some unusual consumption patterns detected - may indicate inefficient appliances"
        else:
            insight = "Multiple anomalies detected - recommend checking for faulty appliances"

        return {
            "patterns": patterns,
            "insight": insight,
            "total_anomalies": len(anomalies),
        }


anomaly_detector = AnomalyDetector()
