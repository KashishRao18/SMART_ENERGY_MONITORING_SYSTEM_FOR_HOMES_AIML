"""
Machine Learning Prediction Module
Energy consumption prediction using Linear Regression and ensemble methods
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance


class EnergyPredictionModel:
    """Machine learning models for energy consumption prediction"""

    def __init__(self):
        self.models = {}
        self.config = {
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
            "target_column": "Global_active_power",
            "datetime_column": "datetime",
            "entity_column": None,
            "frequency": "H",
        }
        self.feature_columns = self.config["feature_columns"]
        self.target_column = self.config["target_column"]
        self.is_trained = False
        self.training_metrics = {}
        self.trained_feature_columns = []
        # Learned defaults/profiles for future feature generation
        self.future_feature_defaults = {
            "Global_reactive_power": 0.1,
            "Global_intensity": 4.0,
        }
        self.hourly_feature_profiles = {}

    def set_config(self, config: dict):
        """Update dataset configuration for dynamic training"""
        if not isinstance(config, dict):
            return
        self.config.update({k: v for k, v in config.items() if v is not None})
        if "feature_columns" in config and config["feature_columns"]:
            self.feature_columns = config["feature_columns"]
        self.target_column = self.config.get("target_column", self.target_column)

    def prepare_features(self, data):
        """Prepare features for model training/prediction"""
        if data is None:
            return None, None

        # Make a copy to avoid modifying the original data
        data = data.copy()

        # Handle datetime column - ensure it's properly parsed
        if "datetime" in data.columns:
            # Convert datetime to proper datetime type if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data["datetime"]):
                try:
                    data["datetime"] = pd.to_datetime(
                        data["datetime"], errors="coerce"
                    )
                except Exception:
                    pass

            # Extract time features only if hour column doesn't exist
            if "hour" not in data.columns:
                # Check if datetime is valid before accessing .dt
                if data[
                    "datetime"
                ].dtype == "object" or not pd.api.types.is_datetime64_any_dtype(
                    data["datetime"]
                ):
                    # Try to convert string datetime to proper format
                    data["datetime"] = pd.to_datetime(
                        data["datetime"], errors="coerce"
                    )

                # Now extract features if datetime is valid
                if pd.api.types.is_datetime64_any_dtype(data["datetime"]):
                    data["hour"] = data["datetime"].dt.hour
                    data["day"] = data["datetime"].dt.day
                    data["month"] = data["datetime"].dt.month
                    data["day_of_week"] = data["datetime"].dt.dayofweek
                    data["is_weekend"] = (
                        data["day_of_week"].isin([5, 6]).astype(int)
                    )
                    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
                    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
                    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
                    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

        available_features = [
            col for col in self.feature_columns if col in data.columns
        ]

        if not available_features:
            return None, None

        # Ensure X is always 2D (required by sklearn)
        X = data[available_features].fillna(0)

        # Convert to numpy array and ensure proper shape
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y = None
        if self.target_column in data.columns:
            y = data[self.target_column].fillna(0)
            # Ensure y is 1D
            if isinstance(y, pd.Series):
                y = y.values
            if isinstance(y, np.ndarray) and y.ndim > 1:
                y = y.ravel()

        return X, y

    def train_models(self, data, tune=False):
        """Train multiple prediction models"""
        X, y = self.prepare_features(data)

        if X is None or y is None:
            return {"error": "Insufficient data for training"}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.models = {
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
        }

        training_metrics = {}

        for name, model in self.models.items():
            if tune and name in ["random_forest", "gradient_boosting"]:
                candidates = []
                if name == "random_forest":
                    param_grid = [
                        {"n_estimators": 100, "max_depth": None},
                        {"n_estimators": 150, "max_depth": 10},
                        {"n_estimators": 200, "max_depth": 12},
                    ]
                    for params in param_grid:
                        m = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
                        m.fit(X_train, y_train)
                        y_pred = m.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        candidates.append((r2, params, m))
                    best = max(candidates, key=lambda x: x[0])
                    model = best[2]
                    self.models[name] = model
                elif name == "gradient_boosting":
                    param_grid = [
                        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
                        {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3},
                        {"n_estimators": 80, "learning_rate": 0.15, "max_depth": 2},
                    ]
                    for params in param_grid:
                        m = GradientBoostingRegressor(random_state=42, **params)
                        m.fit(X_train, y_train)
                        y_pred = m.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        candidates.append((r2, params, m))
                    best = max(candidates, key=lambda x: x[0])
                    model = best[2]
                    self.models[name] = model
                else:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            training_metrics[name] = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2_score": float(r2),
            }

            if (
                "best_model" not in training_metrics
                or r2 > training_metrics["best_model"]["r2_score"]
            ):
                training_metrics["best_model"] = {
                    "name": name,
                    "r2_score": float(r2),
                    "rmse": float(rmse),
                }

        self.training_metrics = training_metrics
        self.is_trained = True
        # Remember the exact columns used during training
        if isinstance(data, pd.DataFrame):
            self.trained_feature_columns = [
                c for c in self.feature_columns if c in data.columns
            ]
            # Learn realistic defaults/profiles for exogenous features used in future prediction
            for col in ["Global_reactive_power", "Global_intensity"]:
                if col in data.columns:
                    try:
                        mean_val = float(pd.to_numeric(data[col], errors="coerce").mean())
                        if np.isfinite(mean_val):
                            self.future_feature_defaults[col] = mean_val
                    except Exception:
                        pass

            if "hour" in data.columns:
                for col in ["Global_reactive_power", "Global_intensity"]:
                    if col in data.columns:
                        try:
                            grp = (
                                data.groupby("hour")[col]
                                .mean()
                                .dropna()
                            )
                            self.hourly_feature_profiles[col] = {
                                int(h): float(v) for h, v in grp.items()
                            }
                        except Exception:
                            self.hourly_feature_profiles[col] = {}
                    else:
                        self.hourly_feature_profiles[col] = {}
        else:
            self.trained_feature_columns = []

        return training_metrics

    def predict(self, data, model_name="random_forest"):
        """Predict energy consumption using trained model"""
        if not self.is_trained:
            return None

        X, _ = self.prepare_features(data)

        # Align to trained feature set to avoid mismatches
        if (
            isinstance(data, pd.DataFrame)
            and self.trained_feature_columns
            and model_name in self.models
        ):
            aligned = data.copy()
            for col in self.trained_feature_columns:
                if col not in aligned.columns:
                    aligned[col] = 0
            aligned = aligned[self.trained_feature_columns]
            X = aligned.values

        if X is None or model_name not in self.models:
            return None

        return self.models[model_name].predict(X)

    def predict_future(self, days=7, model_name="random_forest"):
        """Predict future energy consumption"""
        if not self.is_trained:
            return None

        from datetime import datetime, timedelta
        freq = (self.config.get("frequency") or "H").upper()
        days = max(1, int(days or 1))
        now = datetime.now()
        if freq == "D":
            start = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            future_dates = [start + timedelta(days=i) for i in range(days)]
        else:
            start = (now + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            )
            future_dates = [
                start + timedelta(hours=i) for i in range(days * 24)
            ]

        # Use learned per-hour profiles when available, fallback to learned/global defaults.
        default_reactive = float(self.future_feature_defaults.get("Global_reactive_power", 0.1))
        default_intensity = float(self.future_feature_defaults.get("Global_intensity", 4.0))
        hours = [d.hour for d in future_dates]
        reactive_profile = self.hourly_feature_profiles.get("Global_reactive_power", {}) or {}
        intensity_profile = self.hourly_feature_profiles.get("Global_intensity", {}) or {}
        reactive_vals = [float(reactive_profile.get(h, default_reactive)) for h in hours]
        intensity_vals = [float(intensity_profile.get(h, default_intensity)) for h in hours]

        # Build future feature frame with required columns
        future_payload = {
            self.config.get("datetime_column", "datetime"): future_dates,
            "day": [d.day for d in future_dates],
            "month": [d.month for d in future_dates],
            "year": [d.year for d in future_dates],
            "day_of_week": [d.weekday() for d in future_dates],
            "is_weekend": [1 if d.weekday() >= 5 else 0 for d in future_dates],
            "month_sin": [np.sin(2 * np.pi * d.month / 12) for d in future_dates],
            "month_cos": [np.cos(2 * np.pi * d.month / 12) for d in future_dates],
            "Global_reactive_power": reactive_vals,
            "Global_intensity": intensity_vals,
        }
        if freq == "H":
            future_payload["hour"] = [d.hour for d in future_dates]
            future_payload["hour_sin"] = [
                np.sin(2 * np.pi * d.hour / 24) for d in future_dates
            ]
            future_payload["hour_cos"] = [
                np.cos(2 * np.pi * d.hour / 24) for d in future_dates
            ]
        future_data = pd.DataFrame(future_payload)

        predictions = self.predict(future_data, model_name)
        
        if predictions is None:
            return None

        results = []
        for i, (date, pred) in enumerate(zip(future_dates, predictions)):
            results.append(
                {
                    "datetime": date.strftime("%Y-%m-%d %H:%M:%S"),
                    "hour": date.hour,
                    "predicted_power": float(max(0.1, pred)),
                    "day_type": (
                        "Weekend" if date.weekday() >= 5 else "Weekday"
                    ),
                }
            )

        return results

    def get_feature_importance(self):
        """Get feature importance from Random Forest model"""
        if "random_forest" not in self.models:
            return None

        model = self.models["random_forest"]
        importance = model.feature_importances_

        cols = (
            self.trained_feature_columns
            if self.trained_feature_columns
            else self.feature_columns
        )

        feature_importance = {}
        for feat, imp in zip(cols, importance):
            feature_importance[feat] = float(imp)

        return feature_importance

    def permutation_importances(self, data, model_name="random_forest", n_repeats=5):
        """Compute permutation importances on given data"""
        if model_name not in self.models:
            return None
        X, y = self.prepare_features(data)
        if X is None or y is None:
            return None
        model = self.models[model_name]
        try:
            result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
            cols = self.trained_feature_columns if self.trained_feature_columns else self.feature_columns
            return {feat: float(imp) for feat, imp in zip(cols, result.importances_mean)}
        except Exception:
            return None

    def get_model_comparison(self):
        """Get comparison of all trained models"""
        if not self.is_trained:
            return None

        comparison = []
        for name, metrics in self.training_metrics.items():
            if name != "best_model":
                comparison.append(
                    {
                        "model": name,
                        "r2_score": metrics["r2_score"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                    }
                )

        return sorted(comparison, key=lambda x: x["r2_score"], reverse=True)


prediction_model = EnergyPredictionModel()
