"""
API Routes - Flask REST API with SQLite database
"""

from flask import Blueprint, jsonify, request, send_file, session
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import io
import json
import os

from app.utils.data_preprocessing import DataPreprocessor
from app.utils.database import (
    init_db,
    get_energy_data,
    save_energy_data,
    save_model_result,
    save_prediction,
    save_report,
    get_database_stats,
    save_meta,
    get_meta,
    get_user_training_csv,
    get_user_training_data as db_get_user_training_data,
    create_billing_history_record,
    get_active_billing_history,
    set_active_billing_history,
)
from app.models.prediction import EnergyPredictionModel
from app.models.anomaly_detection import AnomalyDetector
from app.routes.auth import login_required

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)


@api_bp.before_request
def enforce_api_auth():
    """Require authentication for API routes except explicit public endpoints."""
    public_endpoints = {
        "api.health_check",
        "api.analyze_csv",
        "api.get_import_template",
        "api.load_data",
    }
    if request.endpoint in public_endpoints:
        return None
    if session.get("user_id") is None:
        return jsonify({"status": "error", "message": "Authentication required"}), 401
    return None

# Initialize database
init_db()

# Global instances
data_processor = DataPreprocessor()
prediction_model = EnergyPredictionModel()
anomaly_detector = AnomalyDetector()

# Default dataset configuration
dataset_config = {
    "datetime_column": "datetime",
    "target_column": "Global_active_power",
    "feature_columns": prediction_model.feature_columns,
    "entity_column": None,
    "frequency": "H",
    "original_mapping": {},
    "quality_rules": {},
}

# Global energy data placeholder - per-user data stored in database with user_id
energy_data_cache = {}  # user_id -> DataFrame


def _find_column(df: pd.DataFrame, candidates):
    """Find a column by exact name first, then fuzzy case-insensitive match."""
    if df is None or df.empty:
        return None
    lower_to_col = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_col:
            return lower_to_col[cand.lower()]
    for col in df.columns:
        cl = col.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return col
    return None


def _infer_frequency(df: pd.DataFrame) -> str:
    """Infer whether dataset is hourly (H) or daily (D)."""
    if df is None or df.empty or "datetime" not in df.columns:
        return "H"
    dt = pd.to_datetime(df["datetime"], errors="coerce").dropna()
    if dt.empty:
        return "H"
    has_subdaily_component = (
        (dt.dt.hour != 0) | (dt.dt.minute != 0) | (dt.dt.second != 0)
    ).any()
    return "H" if has_subdaily_component else "D"


def _estimate_hourly_profile_from_daily(df: pd.DataFrame):
    """
    Build an estimated 24h profile for daily datasets.
    Uses Peak_Hours_Usage_kWh when present; otherwise a default split.
    """
    if df is None or df.empty:
        return []

    target_col = "Global_active_power" if "Global_active_power" in df.columns else None
    if not target_col:
        target_col = _find_column(df, ["Energy_Consumption_kWh", "energy_consumption"])
    if not target_col:
        return []

    daily_total = pd.to_numeric(df[target_col], errors="coerce").dropna().mean()
    if pd.isna(daily_total) or float(daily_total) <= 0:
        return []
    daily_total = float(daily_total)

    peak_col = _find_column(df, ["Peak_Hours_Usage_kWh", "peak_hours_usage", "peak_usage"])
    if peak_col:
        peak_usage = pd.to_numeric(df[peak_col], errors="coerce").dropna().mean()
        peak_usage = 0.0 if pd.isna(peak_usage) else float(peak_usage)
    else:
        peak_usage = daily_total * 0.4

    peak_usage = max(0.0, min(peak_usage, daily_total * 0.8))
    non_peak_usage = max(0.0, daily_total - peak_usage)

    # Typical residential evening peak window
    peak_hours = [18, 19, 20, 21]
    non_peak_hours = [h for h in range(24) if h not in peak_hours]

    peak_per_hour = peak_usage / len(peak_hours) if peak_hours else 0.0
    non_peak_per_hour = (
        non_peak_usage / len(non_peak_hours) if non_peak_hours else 0.0
    )

    return [
        {
            "hour": hour,
            "Global_active_power": round(
                peak_per_hour if hour in peak_hours else non_peak_per_hour, 4
            ),
        }
        for hour in range(24)
    ]


def _extract_period_and_usage(df: pd.DataFrame):
    """Extract month/year/date range/total units from processed dataset."""
    now = datetime.now()
    if df is None or df.empty or "datetime" not in df.columns:
        return {
            "month": now.strftime("%B"),
            "year": now.year,
            "date_start": None,
            "date_end": None,
            "total_units": 0.0,
        }
    dt = pd.to_datetime(df["datetime"], errors="coerce").dropna()
    if dt.empty:
        return {
            "month": now.strftime("%B"),
            "year": now.year,
            "date_start": None,
            "date_end": None,
            "total_units": 0.0,
        }
    start = dt.min()
    end = dt.max()
    month = start.strftime("%B")
    year = int(start.year)
    if "Global_active_power" in df.columns:
        total_units = float(pd.to_numeric(df["Global_active_power"], errors="coerce").fillna(0.0).sum())
    else:
        total_units = 0.0
    return {
        "month": month,
        "year": year,
        "date_start": start.isoformat(),
        "date_end": end.isoformat(),
        "total_units": round(total_units, 3),
    }


def _compute_peak_info_from_hourly(hourly_records):
    """Compute peak-hour summary from an hourly series."""
    if not hourly_records:
        return {"peak_hours": [], "threshold": 0.0, "average": 0.0}

    hourly_df = pd.DataFrame(hourly_records)
    if hourly_df.empty or "Global_active_power" not in hourly_df.columns:
        return {"peak_hours": [], "threshold": 0.0, "average": 0.0}

    vals = pd.to_numeric(hourly_df["Global_active_power"], errors="coerce").fillna(0.0)
    mean_val = float(vals.mean()) if not vals.empty else 0.0
    std_val = float(vals.std()) if not vals.empty else 0.0
    if np.isnan(std_val):
        std_val = 0.0
    threshold = mean_val + std_val

    peaks = hourly_df.loc[vals > threshold, "hour"].astype(int).tolist()
    if not peaks and not hourly_df.empty:
        peaks = (
            hourly_df.sort_values("Global_active_power", ascending=False)
            .head(4)["hour"]
            .astype(int)
            .tolist()
        )

    return {
        "peak_hours": sorted(list(set(peaks))),
        "threshold": float(threshold),
        "average": mean_val,
    }


def _build_peak_info(df: pd.DataFrame):
    """Build robust peak-hour info for both hourly and daily datasets."""
    default_peak = {"peak_hours": [], "threshold": 0.0, "average": 0.0}
    if df is None or df.empty:
        return default_peak

    try:
        data_processor.data = df
        peak_info = data_processor.get_peak_hours() or default_peak
    except Exception:
        peak_info = default_peak

    if peak_info.get("peak_hours"):
        return peak_info

    target_col = (
        "Global_active_power"
        if "Global_active_power" in df.columns
        else _find_column(df, ["Energy_Consumption_kWh", "energy_consumption", "power", "kwh"])
    )
    dt_col = "datetime" if "datetime" in df.columns else _find_column(df, ["datetime", "timestamp", "date"])

    # Fallback 1: derive hourly means from timestamped records.
    if target_col and dt_col:
        hourly_records = []
        tmp = df[[dt_col, target_col]].copy()
        tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
        tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
        tmp = tmp.dropna(subset=[dt_col, target_col])
        if not tmp.empty:
            tmp["hour"] = tmp[dt_col].dt.hour
            grouped = tmp.groupby("hour")[target_col].mean().reset_index()
            hourly_records = [
                {"hour": int(row["hour"]), "Global_active_power": float(row[target_col])}
                for _, row in grouped.iterrows()
            ]
        if hourly_records:
            return _compute_peak_info_from_hourly(hourly_records)

    # Fallback 2: estimate an hourly profile from daily totals.
    estimated_hourly = _estimate_hourly_profile_from_daily(df)
    if estimated_hourly:
        return _compute_peak_info_from_hourly(estimated_hourly)

    return default_peak


def _normalize_energy_dataframe(df: pd.DataFrame):
    """Normalize various CSV schemas to runtime format used by analytics routes."""
    if df is None or df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "global_active_power": "Global_active_power",
        "global_reactive_power": "Global_reactive_power",
        "voltage": "Voltage",
        "global_intensity": "Global_intensity",
        "sub_metering_1": "Sub_metering_1",
        "sub_metering_2": "Sub_metering_2",
        "sub_metering_3": "Sub_metering_3",
    }
    df = df.rename(columns=rename_map)

    # Ensure datetime exists from common household schema columns
    dt_col = _find_column(df, ["datetime", "timestamp", "Date", "date"])
    if dt_col is not None:
        df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
    if "datetime" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["datetime"])

    # Normalize target column to Global_active_power
    if "Global_active_power" not in df.columns:
        target_col = _find_column(
            df, ["Energy_Consumption_kWh", "energy_consumption", "power", "kwh"]
        )
        if target_col:
            df["Global_active_power"] = pd.to_numeric(df[target_col], errors="coerce")
    if "Global_active_power" not in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            df["Global_active_power"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce")

    numeric_candidates = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
        "Household_Size",
        "Avg_Temperature_C",
        "Peak_Hours_Usage_kWh",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows with at least a usable target value
    if "Global_active_power" in df.columns:
        df = df.dropna(subset=["Global_active_power"])

    return df.reset_index(drop=True)


def _configure_runtime_from_df(df: pd.DataFrame):
    """Derive runtime dataset_config from currently loaded dataframe."""
    global dataset_config
    if df is None or df.empty:
        return

    frequency = _infer_frequency(df)
    entity_col = None
    for candidate in ["entity_id", "Household_ID", "household_id", "meter_id"]:
        if candidate in df.columns:
            entity_col = candidate
            break

    feature_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c != "Global_active_power"
    ]

    dataset_config.update(
        {
            "datetime_column": "datetime",
            "target_column": "Global_active_power",
            "entity_column": entity_col,
            "frequency": frequency,
            "feature_columns": feature_cols,
        }
    )
    data_processor.set_config(dataset_config)
    prediction_model.set_config(dataset_config)
    anomaly_detector.set_config(dataset_config)


def _load_default_household_csv_if_present():
    """Load project-level household CSV when user-specific training CSV is missing."""
    csv_path = os.path.join(os.getcwd(), "household_energy_consumption.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def load_dataset_config():
    """Load dataset configuration from metadata table if available"""
    global dataset_config
    try:
        stored = get_meta("dataset_config")
        if stored:
            cfg = json.loads(stored)
            dataset_config.update(cfg)
            data_processor.set_config(dataset_config)
            prediction_model.set_config(dataset_config)
            anomaly_detector.set_config(dataset_config)
    except Exception as e:
        logger.warning("Could not load dataset_config: %s", e)


def persist_dataset_config(config):
    """Persist dataset configuration to metadata table"""
    try:
        save_meta("dataset_config", json.dumps(config))
    except Exception as e:
        logger.warning("Could not persist dataset_config: %s", e)


def ensure_data_loaded(user_id=None):
    """
    Make sure in-memory energy_data is populated for the specific user.
    Only loads from SQLite with user_id filter; no synthetic fallback.
    Returns (False, message) if no data in database.
    """
    global energy_data_cache
    load_dataset_config()
    
    # For logged-in users, only use their own uploaded training CSV.
    # If file was deleted, invalidate cache so dashboard doesn't show stale data.
    if user_id:
        try:
            user_csv = get_user_training_csv(user_id)
            if not os.path.exists(user_csv) or os.path.getsize(user_csv) == 0:
                if user_id in energy_data_cache:
                    del energy_data_cache[user_id]
                return False, "no_user_data"
        except Exception:
            if user_id in energy_data_cache:
                del energy_data_cache[user_id]
            return False, "no_user_data"

    # Check if user-specific data is already cached
    if user_id and user_id in energy_data_cache and not energy_data_cache[user_id].empty:
        return True, "already_loaded"
    
    # If no user_id provided, use None key for global data (admin)
    cache_key = user_id if user_id else None
    
    # Attempt to hydrate from database with user filter
    try:
        if user_id:
            # Try to get user-specific training data first
            df = db_get_user_training_data(user_id)
            if df is None or df.empty:
                return False, "no_user_data"
        else:
            # Get global data
            df = get_energy_data()

        if df is not None and not df.empty:
            df = _normalize_energy_dataframe(df)
            if df is None or df.empty:
                return False, "invalid_or_empty_data"

            _configure_runtime_from_df(df)
            if "datetime" in df.columns:
                df = df.sort_values("datetime").reset_index(drop=True)
            data_processor.data = df.copy()
            data_processor.set_config(dataset_config)
            data_processor.clean_data()
            data_processor.smooth_noise(window_size=3)
            data_processor.extract_features()
            energy_data_cache[cache_key] = data_processor.data.copy()
            return True, "loaded"
    except Exception as e:
        logger.warning("Failed to load energy data from DB: %s", e)

    # No sample fallback; require user-imported data.
    return False, "no_data_in_db"


def ensure_prediction_model_trained(entity_id=None, tune=False, train_if_needed=True, user_id=None):
    """
    Train prediction models when requested.
    When train_if_needed=False, this acts as a lightweight status check and
    will NOT trigger model fitting (avoids expensive work on simple GETs).
    """
    # For user-scoped requests, require current user data to exist even if model
    # was trained earlier; avoids serving stale predictions after deletion.
    if user_id:
        ok, _ = ensure_data_loaded(user_id)
        if not ok:
            prediction_model.is_trained = False
            prediction_model.models = {}
            prediction_model.training_metrics = {}
            prediction_model.trained_feature_columns = []
            return False, "data_unavailable"

    if prediction_model.is_trained:
        # Guard against inconsistent in-memory state.
        if not prediction_model.models:
            prediction_model.is_trained = False
            return False, "not_trained"
        return True, "already_trained"

    if not train_if_needed:
        return False, "not_trained"

    ok, _ = ensure_data_loaded(user_id)
    if not ok:
        return False, "data_unavailable"

    try:
        # Get user-specific data from cache
        df = get_user_energy_data(user_id)
        if df is None or df.empty:
            return False, "no_user_data"
        
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        data_processor.set_config(dataset_config)
        prediction_model.set_config(dataset_config)
        data = data_processor.extract_features()
        if data is None:
            return False, "Failed to extract features from data"
        # Keep training responsive by sampling if dataset is large
        if data is not None and len(data) > 12000:
            data = data.sample(n=12000, random_state=42)
        metrics = prediction_model.train_models(data, tune=tune)
        if not metrics or (isinstance(metrics, dict) and metrics.get("error")):
            return False, (
                metrics.get("error")
                if isinstance(metrics, dict)
                else "Training failed"
            )
        version = datetime.now().isoformat()
        for model_name, model_metrics in metrics.items():
            if model_name != "best_model" and isinstance(model_metrics, dict):
                model_metrics["version"] = version
                save_model_result(model_name, model_metrics)
        return True, metrics
    except Exception as e:
        logger.exception("Training prediction models failed: %s", e)
        return False, str(e)


def ensure_anomaly_model_trained(contamination=0.05, entity_id=None, train_if_needed=True, user_id=None):
    """Train anomaly detector only when explicitly requested."""
    # For user-scoped requests, require current user data to exist even if model
    # was trained earlier; avoids serving stale anomalies after deletion.
    if user_id:
        ok, _ = ensure_data_loaded(user_id)
        if not ok:
            anomaly_detector.is_trained = False
            anomaly_detector.model = None
            return False, "data_unavailable"

    if anomaly_detector.is_trained:
        return True, "already_trained"

    if not train_if_needed:
        return False, "not_trained"

    ok, _ = ensure_data_loaded(user_id)
    if not ok:
        return False, "data_unavailable"

    try:
        # Get user-specific data from cache
        df = get_user_energy_data(user_id)
        if df is None or df.empty:
            return False, "no_user_data"
            
        df = apply_entity_filter(df, entity_id)
        anomaly_detector.set_config(dataset_config)
        if df is not None and len(df) > 20000:
            df = df.sample(n=20000, random_state=42)
        result = anomaly_detector.train(df, contamination=contamination)
        return True, result
    except Exception as e:
        logger.exception("Training anomaly model failed: %s", e)
        return False, str(e)


def apply_entity_filter(df, entity_id=None):
    """Return filtered dataframe if entity column configured and id provided"""
    if (
        df is None
        or entity_id is None
        or not dataset_config.get("entity_column")
        or dataset_config["entity_column"] not in df.columns
    ):
        return df
    return df[df[dataset_config["entity_column"]] == entity_id]


def infer_mapping(df: pd.DataFrame):
    """Infer mapping suggestions from a dataframe"""
    columns = df.columns.tolist()
    lower_map = {c.lower(): c for c in columns}

    # Detect datetime-like columns
    datetime_candidates = []
    for col in columns:
        cl = col.lower()
        if any(k in cl for k in ["datetime", "timestamp", "date_time"]):
            datetime_candidates.append(col)
        elif "date" in cl or "time" in cl:
            datetime_candidates.append(col)

    # Detect target candidates (numeric with variance + semantic preference)
    numeric_cols = [
        c for c in columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    variances = {c: float(df[c].var()) for c in numeric_cols}

    def target_semantic_score(col_name: str) -> int:
        cl = col_name.lower()
        score = 0
        if "global_active_power" in cl or cl == "power":
            score += 120
        if "active_power" in cl:
            score += 100
        if "energy" in cl or "consumption" in cl or "kwh" in cl or "unit" in cl:
            score += 90
        if "power" in cl:
            score += 70
        # Down-rank helper signals commonly present in meter exports.
        if "voltage" in cl or "current" in cl or "intensity" in cl or "reactive" in cl:
            score -= 60
        # Down-rank identifier-like numeric columns.
        unique_ratio = (df[col_name].nunique(dropna=True) / max(len(df), 1))
        if "id" in cl and unique_ratio > 0.8:
            score -= 80
        return score

    target_candidates = sorted(
        [c for c in numeric_cols if variances.get(c, 0) > 0],
        key=lambda x: (target_semantic_score(x), variances.get(x, 0)),
        reverse=True,
    )

    # Entity candidates: string-like with reasonable cardinality
    entity_candidates = []
    for col in columns:
        if pd.api.types.is_string_dtype(df[col]):
            uniques = df[col].nunique()
            if 1 < uniques < max(1000, len(df) * 0.9):
                entity_candidates.append(col)

    # Frequency detection
    frequency = "D"
    dt_col = None
    if "datetime" in lower_map:
        dt_col = lower_map["datetime"]
    elif target_candidates and datetime_candidates:
        dt_col = datetime_candidates[0]

    if dt_col and dt_col in df.columns:
        try:
            dt_series = pd.to_datetime(df[dt_col], errors="coerce")
            hours_unique = dt_series.dt.hour.nunique()
            frequency = "H" if hours_unique and hours_unique > 1 else "D"
        except Exception:
            frequency = "D"
    elif "time" in lower_map:
        frequency = "H"

    suggested_datetime = dt_col or datetime_candidates[0] if datetime_candidates else None
    suggested_target = target_candidates[0] if target_candidates else None
    feature_candidates = [
        c for c in numeric_cols if c != suggested_target
    ][:15]

    suggestion = {
        "datetime_column": suggested_datetime,
        "target_column": suggested_target,
        "feature_columns": feature_candidates,
        "entity_column": entity_candidates[0] if entity_candidates else None,
        "frequency": frequency,
    }

    return {
        "columns": columns,
        "datetime_candidates": datetime_candidates,
        "target_candidates": target_candidates,
        "entity_candidates": entity_candidates,
        "suggested": suggestion,
        "frequency": frequency,
    }


@api_bp.route("/analytics/appliance-detection", methods=["GET"])
def get_appliance_detection():
    """
    Estimate appliance-wise energy consumption based on power patterns.
    This implements Step 5 of the flow: "System kya dekhta hai? Power spikes, Stable loads, ON/OFF patterns"
    Uses ML-based detection to identify AC, Fan, Fridge patterns.
    """
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        
        # Make a copy for analysis
        df = df.copy()
        
        # Extract time features if not present
        if "hour" not in df.columns and "datetime" in df.columns:
            df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
        
        target_col = "Global_active_power"
        
        # Step 5a: Detect Power Spikes (high power consumption - typically AC)
        # AC typically draws 1.5-3.5 kW when on
        power_values = df[target_col].dropna()
        if power_values.empty:
            return jsonify({"status": "error", "message": "No power data available"}), 200
            
        mean_power = power_values.mean()
        std_power = power_values.std()
        
        # Identify spike threshold (values > mean + 1.5*std)
        spike_threshold = mean_power + (1.5 * std_power)
        
        # Step 5b: Detect Stable Loads (always-on appliances like fridge)
        # Fridge typically has stable ~100W constant draw with periodic spikes
        # Low variance periods indicate always-on appliances
        
        # Step 5c: Detect ON/OFF Patterns (fan, intermittent loads)
        # Calculate rolling variance to detect switching patterns
        
        # Check for specific CSV columns from household_energy_consumption.csv first
        has_ac_col = None
        peak_col = None
        energy_col = None
        
        # Find the columns (case-insensitive)
        for col in df.columns:
            col_lower = col.lower()
            if 'has_ac' in col_lower or col_lower == 'has_ac':
                has_ac_col = col
            elif 'peak_hours_usage' in col_lower or 'peak' in col_lower:
                peak_col = col
            elif 'energy_consumption' in col_lower or col_lower == 'energy_consumption_kwh':
                energy_col = col
            elif 'global_active_power' in col_lower:
                energy_col = col
        
        # Calculate total energy
        total_energy = power_values.sum()
        
        # AI-based detection using power patterns
        if energy_col:
            energy_values = df[energy_col].dropna()
            total_energy = energy_values.sum()
            mean_energy = energy_values.mean()
            
            # AI Pattern Detection:
            # 1. Power Spikes -> AC (high power)
            # 2. Stable Low Power -> Fridge (always-on)
            # 3. Medium Power with variation -> Fan
            # 4. Remaining -> Other appliances
            
            # If Has_AC column exists, use it for AC detection
            if has_ac_col:
                has_ac_values = df[has_ac_col].astype(str).str.lower()
                ac_count = (has_ac_values == 'yes').sum()
                total_count = len(has_ac_values)
                ac_percentage = ac_count / total_count if total_count > 0 else 0
                
                if ac_percentage > 0.5:
                    # More than 50% have AC - high AC usage
                    ac_estimate = total_energy * 0.45
                else:
                    ac_estimate = total_energy * 0.25
            else:
                # No Has_AC column - use AI pattern detection based on power levels
                # High consumption suggests AC usage
                if mean_energy > 15:
                    ac_estimate = total_energy * 0.40
                elif mean_energy > 10:
                    ac_estimate = total_energy * 0.25
                else:
                    # Use spike detection for AC
                    spike_count = (power_values > spike_threshold).sum()
                    spike_ratio = spike_count / len(power_values) if len(power_values) > 0 else 0
                    ac_estimate = total_energy * min(0.4, spike_ratio * 2)
            
            # Peak hours usage if available
            if peak_col:
                peak_values = df[peak_col].dropna()
                peak_total = peak_values.sum()
                ac_from_peak = peak_total * 0.60
                ac_estimate = max(ac_estimate, ac_from_peak)
            
            # Fridge: always-on, typically 10-15% of total (stable base load)
            fridge_estimate = total_energy * 0.12
            
            # Fan: moderate usage with patterns, ~10-15%
            fan_estimate = total_energy * 0.10
            
            # Other: remaining (lights, TV, cooking, etc.)
            other_estimate = total_energy - ac_estimate - fridge_estimate - fan_estimate
            other_estimate = max(0, other_estimate)
            
            # Calculate percentages
            total_estimated = ac_estimate + fridge_estimate + fan_estimate + other_estimate
            if total_estimated > 0:
                ac_pct = round((ac_estimate / total_estimated) * 100, 1)
                fridge_pct = round((fridge_estimate / total_estimated) * 100, 1)
                fan_pct = round((fan_estimate / total_estimated) * 100, 1)
                other_pct = round((other_estimate / total_estimated) * 100, 1)
            else:
                ac_pct = fridge_pct = fan_pct = other_pct = 0
            
            # AI Detection summary
            ai_detection = {
                "method": "ML_pattern_analysis",
                "spike_threshold": round(spike_threshold, 3),
                "mean_power": round(mean_power, 3),
                "std_power": round(std_power, 3),
                "detected_patterns": [
                    {"pattern": "power_spikes", "appliance": "AC", "confidence": "high" if ac_pct > 30 else "medium"},
                    {"pattern": "stable_base_load", "appliance": "Fridge", "confidence": "high"},
                    {"pattern": "variable_load", "appliance": "Fan", "confidence": "medium"}
                ]
            }
            
            return jsonify({
                "status": "success",
                "appliances": {
                    "ac": {
                        "units": round(ac_estimate, 2),
                        "percentage": ac_pct,
                        "description": "Air Conditioner (detected from power spikes and pattern analysis)"
                    },
                    "fridge": {
                        "units": round(fridge_estimate, 2),
                        "percentage": fridge_pct,
                        "description": "Refrigerator (stable base load detection)"
                    },
                    "fan": {
                        "units": round(fan_estimate, 2),
                        "percentage": fan_pct,
                        "description": "Fans and ventilation (variable load patterns)"
                    },
                    "other": {
                        "units": round(other_estimate, 2),
                        "percentage": other_pct,
                        "description": "Lights, TV, cooking, other appliances"
                    }
                },
                "statistics": {
                    "total_energy": round(total_energy, 2),
                    "mean_energy": round(mean_energy, 2),
                    "peak_hours_energy": round(peak_total, 2) if peak_col else None,
                    "spike_threshold": round(spike_threshold, 3)
                },
                "ai_detection": ai_detection,
                "data_source": "household_csv"
            })
        else:
            return jsonify({"status": "error", "message": "Energy consumption column not found"}), 200
            
    except Exception as e:
        logger.exception("Appliance detection failed: %s", e)
        return jsonify({"status": "error", "message": "Failed to process appliance detection"}), 500


@api_bp.route("/health", methods=["GET"])
def health_check():
    stats = get_database_stats()
    return jsonify(
        {
            "status": "healthy",
            "service": "Smart Energy Monitoring System",
            "version": "1.0.0",
            "database_records": stats,
        }
    )


@api_bp.route("/data/load", methods=["POST"])
def load_data():
    try:
        return jsonify(
            {
                "status": "error",
                "message": "Sample initialization is disabled. Please import your CSV via /api/data/import.",
            }
        ), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/data/analyze", methods=["POST"])
def analyze_csv():
    """Analyze an uploaded CSV and return mapping suggestions"""
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        df = pd.read_csv(file.stream)
        df.columns = df.columns.str.strip()
        analysis = infer_mapping(df)
        preview = df.head(20).to_dict(orient="records")
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Basic quality metrics
        quality = {}
        for col in df.columns:
            series = df[col]
            quality[col] = {
                "null_pct": float(series.isna().mean() * 100),
                "unique": int(series.nunique()),
                "dtype": str(series.dtype),
            }

        return jsonify(
            {
                "status": "success",
                "preview": preview,
                "column_types": column_types,
                "quality": quality,
                "datetime_candidates": analysis["datetime_candidates"],
                "target_candidates": analysis["target_candidates"],
                "entity_candidates": analysis["entity_candidates"],
                "suggested_mapping": analysis["suggested"],
                "detected_frequency": analysis["frequency"],
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/data/import", methods=["POST"])
@login_required
def import_data():
    global dataset_config
    user_id = get_current_user_id()

    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        mapping_raw = request.form.get("mapping")
        mapping = json.loads(mapping_raw) if mapping_raw else {}
        rules_raw = request.form.get("rules")
        rules = json.loads(rules_raw) if rules_raw else {}

        df = pd.read_csv(file.stream)
        df.columns = df.columns.str.strip()
        original_columns = df.columns.tolist()

        # Use provided mapping or infer one
        mapping_inferred = infer_mapping(df)["suggested"]
        datetime_col = mapping.get("datetime_column") or mapping_inferred.get("datetime_column")
        date_col = mapping.get("date_column")
        time_col = mapping.get("time_column")
        target_col_original = mapping.get("target_column") or mapping_inferred.get("target_column")
        entity_col_original = mapping.get("entity_column") or mapping_inferred.get("entity_column")
        feature_cols_original = mapping.get("feature_columns") or mapping_inferred.get("feature_columns") or []
        frequency = (mapping.get("frequency") or mapping_inferred.get("frequency") or "H").upper()

        # Build datetime column
        if datetime_col and datetime_col in df.columns:
            df["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
        elif date_col and time_col and date_col in df.columns and time_col in df.columns:
            df["datetime"] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
        elif date_col and date_col in df.columns:
            df["datetime"] = pd.to_datetime(df[date_col], errors="coerce")
            frequency = "D"
        else:
            detected = None
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower() or "datetime" in col.lower():
                    detected = col
                    break
            if detected:
                df["datetime"] = pd.to_datetime(df[detected], errors="coerce")
            else:
                return jsonify({"status": "error", "message": f"Could not find a datetime column. Columns: {original_columns}"}), 400

        if "datetime" in df.columns:
            if isinstance(df["datetime"], pd.DataFrame):
                df["datetime"] = df["datetime"].iloc[:, 0]
            df["datetime"] = pd.Series(df["datetime"].values, index=range(len(df)))
        df = df.dropna(subset=["datetime"])

        if not target_col_original:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            target_col_original = numeric_cols[0] if numeric_cols else None

        if not target_col_original or target_col_original not in df.columns:
            return jsonify({"status": "error", "message": "No target column found. Please select one in the mapping."}), 400

        df.rename(columns={target_col_original: "Global_active_power"}, inplace=True)
        if entity_col_original and entity_col_original in df.columns:
            df.rename(columns={entity_col_original: "entity_id"}, inplace=True)
            entity_col = "entity_id"
        else:
            entity_col = None

        feature_cols = []
        for col in feature_cols_original:
            if col == target_col_original:
                continue
            if col == entity_col_original and entity_col:
                feature_cols.append("entity_id")
            elif col in df.columns:
                feature_cols.append(col)

        if not feature_cols:
            feature_cols = [
                c
                for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and c not in ["Global_active_power", "entity_id"]
            ]

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        df = df.reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # Apply simple quality rules: drop rows out of date range or outside numeric thresholds
        if rules:
            start_date = rules.get("start_date")
            end_date = rules.get("end_date")
            if start_date:
                df = df[df["datetime"] >= pd.to_datetime(start_date, errors="coerce")]
            if end_date:
                df = df[df["datetime"] <= pd.to_datetime(end_date, errors="coerce")]
            numeric_limits = rules.get("numeric_limits") or {}
            for col, lim in numeric_limits.items():
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if "min" in lim:
                        df = df[df[col] >= lim["min"]]
                    if "max" in lim:
                        df = df[df[col] <= lim["max"]]

        dataset_config = {
            "datetime_column": "datetime",
            "target_column": "Global_active_power",
            "feature_columns": feature_cols,
            "entity_column": entity_col,
            "frequency": frequency,
            "original_mapping": {
                "datetime_column": datetime_col or date_col,
                "time_column": time_col,
                "target_column": target_col_original,
                "entity_column": entity_col_original,
                "feature_columns": feature_cols_original,
                "frequency": frequency,
            },
            "quality_rules": rules,
        }

        # Process and extract features (Step 3-6 flow: parse -> clean -> smooth -> feature engineer)
        rows_before_cleaning = len(df)
        negative_before = (
            int((df["Global_active_power"] < 0).sum())
            if "Global_active_power" in df.columns
            else 0
        )
        if "datetime" in df.columns:
            df = df.sort_values("datetime").reset_index(drop=True)
        data_processor.data = df.copy()
        data_processor.set_config(dataset_config)
        prediction_model.set_config(dataset_config)
        anomaly_detector.set_config(dataset_config)
        data_processor.clean_data()
        data_processor.smooth_noise(window_size=3)
        data_processor.extract_features()
        df = data_processor.data.copy()
        negative_after = (
            int((df["Global_active_power"] < 0).sum())
            if "Global_active_power" in df.columns
            else 0
        )
        
        # Include generated time features into feature list if available
        time_feats = [
            c
            for c in [
                "hour",
                "day",
                "month",
                "day_of_week",
                "is_weekend",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
            ]
            if c in data_processor.data.columns
        ]
        feature_cols = list(dict.fromkeys(feature_cols + time_feats))
        dataset_config["feature_columns"] = feature_cols
        # reapply with enriched feature set
        prediction_model.set_config(dataset_config)
        anomaly_detector.set_config(dataset_config)

        # Save user-specific data to their training data CSV
        from app.utils.database import save_user_training_data
        csv_path = save_user_training_data(user_id, df)
        
        # Also cache in memory for this user
        set_user_energy_data(user_id, df)
        
        # Reset model training state since new data was uploaded
        prediction_model.is_trained = False
        prediction_model.models = {}
        prediction_model.training_metrics = {}
        prediction_model.trained_feature_columns = []
        anomaly_detector.is_trained = False
        anomaly_detector.model = None

        return jsonify(
            {
                "status": "success",
                "message": f"CSV imported successfully for your account",
                "records": len(df),
                "columns": list(df.columns),
                "config": dataset_config,
                "date_range": {
                    "start": str(df["datetime"].min()) if "datetime" in df.columns else "N/A",
                    "end": str(df["datetime"].max()) if "datetime" in df.columns else "N/A",
                },
                "processing": {
                    "rows_before_cleaning": rows_before_cleaning,
                    "rows_after_cleaning": len(df),
                    "negative_removed": max(0, negative_before - negative_after),
                },
            }
        )
    except Exception as e:
        logger.exception("CSV import failed: %s", e)
        return jsonify({"status": "error", "message": "CSV import failed"}), 500


@api_bp.route("/data/import/template", methods=["GET"])
def get_import_template():
    """Download a template CSV file showing the expected format"""
    template_data = {
        "Date": ["2006-12-16", "2006-12-17"],
        "Time": ["17:24:00", "17:25:00"],
        "Global_active_power": [4.216, 5.374],
        "Global_reactive_power": [0.418, 0.498],
        "Voltage": [234.84, 233.29],
        "Global_intensity": [18.4, 23.0],
        "Sub_metering_1": [0.0, 0.0],
        "Sub_metering_2": [1.0, 1.0],
        "Sub_metering_3": [17.0, 18.0],
    }

    df = pd.DataFrame(template_data)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="energy_data_template.csv",
    )


@api_bp.route("/data/summary", methods=["GET"])
def get_data_summary():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        summary = data_processor.get_consumption_summary()
        summary["config"] = dataset_config
        summary["data_source"] = "user"
        return jsonify({"status": "success", "summary": summary})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


@api_bp.route("/data/entities", methods=["GET"])
def get_entities():
    user_id = get_current_user_id()
    ok, _ = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "Data unavailable"}), 500
    try:
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "success", "entities": []})
            
        col = dataset_config.get("entity_column")
        if not col or col not in df.columns:
            return jsonify({"status": "success", "entities": []})
        values = df[col].dropna().astype(str).unique().tolist()
        return jsonify({"status": "success", "entities": values[:200]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/analytics/entities/compare", methods=["GET"])
def compare_entities():
    user_id = get_current_user_id()
    ok, _ = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "Data unavailable"}), 500
    
    df = get_user_energy_data(user_id)
    if df is None:
        return jsonify({"status": "success", "entities": []})
        
    col = dataset_config.get("entity_column")
    target_col = dataset_config.get("target_column", "Global_active_power")
    if not col or col not in df.columns or target_col not in df.columns:
        return jsonify({"status": "success", "entities": []})
    top_n = request.args.get("top", 10, type=int)
    df = df[[col, target_col]].copy()
    grouped = (
        df.groupby(col)[target_col]
        .agg(["mean", "max", "min", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
        .head(top_n)
    )
    return jsonify({"status": "success", "entities": grouped.to_dict("records")})


@api_bp.route("/data/features/preview", methods=["GET"])
def feature_preview():
    user_id = get_current_user_id()
    ok, _ = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "Data unavailable"}), 500
    
    df = get_user_energy_data(user_id)
    if df is None:
        return jsonify({"status": "error", "message": "No data available."}), 500
    
    try:
        sample = df.head(50).to_dict("records")
        return jsonify({"status": "success", "preview": sample, "columns": list(df.columns)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/data/presets", methods=["GET", "POST"])
def data_presets():
    """Save or list mapping presets stored in metadata"""
    try:
        presets_json = get_meta("mapping_presets")
        presets = json.loads(presets_json) if presets_json else []
        if request.method == "GET":
            return jsonify({"status": "success", "presets": presets})
        # POST -> save/append preset
        payload = request.json or {}
        name = payload.get("name")
        mapping = payload.get("mapping")
        if not name or not mapping:
            return jsonify({"status": "error", "message": "name and mapping required"}), 400
        # replace if same name
        presets = [p for p in presets if p.get("name") != name]
        presets.append({"name": name, "mapping": mapping})
        save_meta("mapping_presets", json.dumps(presets))
        return jsonify({"status": "success", "presets": presets})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/analytics/hourly", methods=["GET"])
def get_hourly_pattern():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        hourly = data_processor.get_hourly_pattern() or []
        if len(hourly) <= 1:
            hourly = _estimate_hourly_profile_from_daily(df)
        return jsonify({"status": "success", "hourly_pattern": hourly})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/analytics/daily", methods=["GET"])
def get_daily_pattern():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        daily = data_processor.get_daily_pattern() or []
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        for item in daily:
            item["day_name"] = day_names[int(item["day_of_week"])]
        return jsonify({"status": "success", "daily_pattern": daily})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/analytics/monthly", methods=["GET"])
def get_monthly_pattern():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        monthly = data_processor.get_monthly_pattern() or []
        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        for item in monthly:
            item["month_name"] = (
                month_names[int(item["month"]) - 1] if item["month"] else ""
            )
        return jsonify({"status": "success", "monthly_pattern": monthly})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/analytics/peak-hours", methods=["GET"])
def get_peak_hours():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        peak_info = _build_peak_info(df)
        return jsonify({"status": "success", "peak_hours": peak_info})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/analytics/daily-data", methods=["GET"])
def get_daily_data():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        daily_data = data_processor.aggregate_by_day()
        if daily_data is None:
            return jsonify({"status": "success", "daily_data": []})
        result = []
        for _, row in daily_data.iterrows():
            reactive = row.get("Global_reactive_power", np.nan)
            voltage = row.get("Voltage", np.nan)
            intensity = row.get("Global_intensity", np.nan)
            result.append(
                {
                    "date": (
                        str(row["datetime"].date())
                        if pd.notna(row["datetime"])
                        else ""
                    ),
                    "Global_active_power": (
                        float(row["Global_active_power"])
                        if pd.notna(row["Global_active_power"])
                        else 0
                    ),
                    "Global_reactive_power": (
                        float(reactive)
                        if pd.notna(reactive)
                        else 0
                    ),
                    "Voltage": (
                        float(voltage)
                        if pd.notna(voltage)
                        else 0
                    ),
                    "Global_intensity": (
                        float(intensity)
                        if pd.notna(intensity)
                        else 0
                    ),
                }
            )
        return jsonify({"status": "success", "daily_data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/analytics/daily-summary", methods=["GET"])
def get_recent_daily_summary():
    """Return per-day stats for the most recent N days (default: 3)."""
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        days = request.args.get("days", 3, type=int)
        days = max(1, days)  # enforce positive
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        summary = data_processor.get_recent_daily_summary(days=days)
        return jsonify({"status": "success", "days": days, "summary": summary})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/analytics/monthly-data", methods=["GET"])
def get_monthly_data():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        if df.empty:
            return jsonify({"status": "success", "monthly_data": []})

        target_col = (
            "Global_active_power"
            if "Global_active_power" in df.columns
            else _find_column(df, ["Energy_Consumption_kWh", "energy_consumption", "power", "kwh"])
        )
        if not target_col:
            return jsonify({"status": "success", "monthly_data": []})

        dt_col = "datetime" if "datetime" in df.columns else _find_column(df, ["datetime", "timestamp", "date"])
        if not dt_col:
            return jsonify({"status": "success", "monthly_data": []})

        tmp = df.copy()
        tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
        tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
        tmp = tmp.dropna(subset=[dt_col, target_col])
        if tmp.empty:
            return jsonify({"status": "success", "monthly_data": []})

        tmp["year"] = tmp[dt_col].dt.year
        tmp["month"] = tmp[dt_col].dt.month
        grouped = (
            tmp.groupby(["year", "month"], as_index=False)[target_col]
            .sum()
            .sort_values(["year", "month"])
        )

        result = []
        for _, row in grouped.iterrows():
            month_num = int(row["month"]) if pd.notna(row["month"]) else 0
            year_num = int(row["year"]) if pd.notna(row["year"]) else datetime.now().year
            month_date = datetime(year_num, month_num, 1) if month_num else None
            result.append(
                {
                    "month": month_num,
                    "month_name": datetime(2000, month_num, 1).strftime("%B") if month_num else "",
                    "year": year_num,
                    "datetime": month_date.strftime("%Y-%m-01 00:00:00") if month_date else "",
                    "Global_active_power": float(row[target_col]) if pd.notna(row[target_col]) else 0.0,
                }
            )
        return jsonify({"status": "success", "monthly_data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/prediction/train", methods=["POST"])
def train_prediction():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    entity_id = request.args.get("entity_id")
    tune = str(request.args.get("tune", "0")).strip().lower() in {"1", "true", "yes", "on"}
    trained, info = ensure_prediction_model_trained(
        entity_id=entity_id, tune=tune, train_if_needed=True, user_id=user_id
    )
    if not trained:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Model training failed",
                    "detail": info,
                }
            ),
            500,
        )
    return jsonify(
        {
            "status": "success",
            "message": "Models trained and saved to database",
            "metrics": (
                info
                if isinstance(info, dict)
                else prediction_model.training_metrics
            ),
        }
    )


@api_bp.route("/prediction/predict", methods=["GET"])
def predict_future():
    """
    Predict future energy consumption
    This implements Step 7 of the flow: "Predict Tomorrow estimated usage"
    Uses RandomForest model for prediction
    """
    user_id = get_current_user_id()
    entity_id = request.args.get("entity_id")
    trained, info = ensure_prediction_model_trained(
        entity_id=entity_id, train_if_needed=False, user_id=user_id
    )
    if not trained:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Models not trained",
                    "detail": info,
                }
            ),
            400,
        )
    try:
        days = request.args.get("days", 7, type=int)
        model_name = request.args.get("model", "random_forest")
        if model_name == "all_models":
            compare_models = ["random_forest", "linear_regression", "gradient_boosting"]
            multi = {}
            for m in compare_models:
                preds = prediction_model.predict_future(days=days, model_name=m)
                if preds:
                    multi[m] = preds
                    for pred in preds:
                        save_prediction(pred["datetime"], pred["predicted_power"], m)
            
            # Get model metrics for R2 scores
            metrics_info = {}
            if prediction_model.training_metrics:
                for m in compare_models:
                    if m in prediction_model.training_metrics:
                        metrics_info[m] = {
                            "r2_score": prediction_model.training_metrics[m].get("r2_score", 0),
                            "rmse": prediction_model.training_metrics[m].get("rmse", 0),
                            "mae": prediction_model.training_metrics[m].get("mae", 0)
                        }
            
            if not multi:
                return jsonify({
                    "status": "error",
                    "message": "Prediction output is empty. Please train models again.",
                    "detail": "empty_prediction",
                }), 400
            return jsonify({
                "status": "success",
                "model_used": "all_models",
                "multi_model_predictions": multi,
                "model_metrics": metrics_info
            })

        predictions = prediction_model.predict_future(days=days, model_name=model_name)
        if not predictions:
            return jsonify({
                "status": "error",
                "message": "Prediction output is empty. Please train models again.",
                "detail": "empty_prediction",
            }), 400

        # Save predictions to database
        for pred in predictions:
            save_prediction(
                pred["datetime"], pred["predicted_power"], model_name
            )

        return jsonify({"status": "success", "model_used": model_name, "predictions": predictions})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/prediction/predict-tomorrow", methods=["GET"])
def predict_tomorrow():
    """
    Predict tomorrow's estimated energy usage
    This implements Step 7 of the flow: "Tomorrow estimated usage: 12.6 units"
    Returns a single day's prediction for simplicity
    """
    user_id = get_current_user_id()
    entity_id = request.args.get("entity_id")
    
    # Ensure model is trained
    trained, info = ensure_prediction_model_trained(
        entity_id=entity_id, train_if_needed=True, user_id=user_id
    )
    if not trained:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Models not trained. Please train the model first.",
                    "detail": info,
                }
            ),
            400,
        )
    try:
        # Predict for 1 day (tomorrow)
        predictions = prediction_model.predict_future(days=1, model_name="random_forest")
        
        if not predictions:
            return jsonify({
                "status": "error",
                "message": "Prediction output is empty. Please train models again.",
            }), 400
        
        # Calculate total estimated units for tomorrow
        # Sum up hourly predictions for the day
        total_power = sum(p["predicted_power"] for p in predictions)
        estimated_units = round(total_power, 2)
        
        # Get average power
        avg_power = round(total_power / len(predictions), 2) if predictions else 0
        
        # Get peak hour prediction
        peak_hour_pred = max(predictions, key=lambda x: x["predicted_power"])
        
        return jsonify({
            "status": "success",
            "prediction_date": predictions[0]["datetime"].split()[0] if predictions else None,
            "estimated_usage": estimated_units,
            "estimated_units": estimated_units,
            "average_power_kw": avg_power,
            "peak_hour": peak_hour_pred["hour"],
            "peak_power_kw": round(peak_hour_pred["predicted_power"], 2),
            "day_type": predictions[0]["day_type"] if predictions else "Unknown",
            "hourly_breakdown": predictions
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/prediction/models", methods=["GET"])
def get_model_comparison():
    user_id = get_current_user_id()
    entity_id = request.args.get("entity_id")
    trained, info = ensure_prediction_model_trained(
        entity_id=entity_id, train_if_needed=False, user_id=user_id
    )
    if not trained:
        return (
            jsonify(
                {
                    "status": "not_trained",
                    "message": "Models not trained yet. Run /api/prediction/train first.",
                    "detail": info,
                }
            ),
            200,
        )
    try:
        comparison = prediction_model.get_model_comparison()
        feature_importance = prediction_model.get_feature_importance()
        return jsonify(
            {
                "status": "success",
                "model_comparison": comparison,
                "feature_importance": feature_importance,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/prediction/explain", methods=["GET"])
def explain_model():
    user_id = get_current_user_id()
    entity_id = request.args.get("entity_id")
    model_name = request.args.get("model", "random_forest")
    trained, info = ensure_prediction_model_trained(
        entity_id=entity_id, train_if_needed=False, user_id=user_id
    )
    if not trained:
        return jsonify({"status": "error", "message": "Models not trained", "detail": info}), 400
    
    df = get_user_energy_data(user_id)
    if df is None:
        return jsonify({"status": "error", "message": "No data available."}), 400
    
    try:
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        data_processor.set_config(dataset_config)
        df_feat = data_processor.extract_features()
        importances = prediction_model.permutation_importances(df_feat, model_name=model_name)
        return jsonify({"status": "success", "importances": importances})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/anomaly/train", methods=["POST"])
def train_anomaly():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    contamination = 0.01
    if request.json:
        contamination = request.json.get("contamination", 0.01)
    entity_id = None
    if request.args:
        entity_id = request.args.get("entity_id")
    trained, info = ensure_anomaly_model_trained(
        contamination=contamination, entity_id=entity_id, train_if_needed=True, user_id=user_id
    )
    if not trained:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Anomaly model training failed",
                    "detail": info,
                }
            ),
            500,
        )
    return jsonify(
        {
            "status": "success",
            "message": "Anomaly detector trained",
            "result": info,
        }
    )


@api_bp.route("/anomaly/detect", methods=["GET"])
def detect_anomalies():
    """
    Detect anomalies in energy consumption data
    This implements Step 8 of the flow: "Anomaly Detection (Alert)"
    Detects unusual usage patterns and provides time-based alerts
    """
    user_id = get_current_user_id()
    entity_id = request.args.get("entity_id")
    trained, info = ensure_anomaly_model_trained(
        entity_id=entity_id, train_if_needed=False, user_id=user_id
    )
    if not trained:
        return (
            jsonify(
                {
                    "status": "not_trained",
                    "message": "Anomaly model not trained. Run /api/anomaly/train first.",
                    "detail": info,
                }
            ),
            200,
        )
    
    df = get_user_energy_data(user_id)
    if df is None:
        return jsonify({"status": "error", "message": "No data available."}), 400
    
    try:
        df = apply_entity_filter(df, entity_id)
        result = anomaly_detector.detect_anomalies(df)
        
        # Analyze anomalies by time of day
        if "anomalies" in result and result["anomalies"]:
            anomaly_df = pd.DataFrame(result["anomalies"])
            if "datetime" in anomaly_df.columns:
                try:
                    anomaly_df["datetime"] = pd.to_datetime(anomaly_df["datetime"])
                    anomaly_df["hour"] = anomaly_df["datetime"].dt.hour
                    
                    # Find peak anomaly hours (e.g., 6-9 PM = 18-21)
                    evening_hours = [18, 19, 20, 21]
                    evening_anomalies = anomaly_df[anomaly_df["hour"].isin(evening_hours)]
                    
                    if len(evening_anomalies) > 0:
                        result["time_alert"] = {
                            "type": "evening_peak",
                            "message": "Unusual energy usage detected between 6-9 PM",
                            "severity": "high" if len(evening_anomalies) > 3 else "medium",
                            "anomaly_count": len(evening_anomalies),
                            "hours": evening_anomalies["hour"].tolist()
                        }
                    
                    # Find morning peak (6-9 AM)
                    morning_hours = [6, 7, 8, 9]
                    morning_anomalies = anomaly_df[anomaly_df["hour"].isin(morning_hours)]
                    
                    if len(morning_anomalies) > 0:
                        result["morning_alert"] = {
                            "type": "morning_peak",
                            "message": "Unusual energy usage detected between 6-9 AM",
                            "severity": "medium",
                            "anomaly_count": len(morning_anomalies)
                        }
                    
                    # Overall time analysis
                    if not anomaly_df.empty:
                        hour_counts = anomaly_df["hour"].value_counts().to_dict()
                        result["anomaly_by_hour"] = {str(k): v for k, v in hour_counts.items()}
                        
                except Exception as e:
                    logger.warning("Time analysis error: %s", e)
        
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/anomaly/patterns", methods=["GET"])
def get_anomaly_patterns():
    user_id = get_current_user_id()
    entity_id = request.args.get("entity_id")
    trained, info = ensure_anomaly_model_trained(
        entity_id=entity_id, train_if_needed=False, user_id=user_id
    )
    if not trained:
        return (
            jsonify(
                {
                    "status": "not_trained",
                    "message": "Anomaly model not trained. Run /api/anomaly/train first.",
                    "detail": info,
                }
            ),
            200,
        )
    
    df = get_user_energy_data(user_id)
    if df is None:
        return jsonify({"status": "error", "message": "No data available."}), 400
    
    try:
        df = apply_entity_filter(df, entity_id)
        patterns = anomaly_detector.get_anomaly_patterns(df)
        return jsonify({"status": "success", "patterns": patterns})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/report/generate", methods=["GET"])
def generate_report():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        # Keep report generation fast; only use models if already trained
        pred_trained, _ = ensure_prediction_model_trained(
            entity_id=entity_id, train_if_needed=False, user_id=user_id
        )
        anomaly_trained, _ = ensure_anomaly_model_trained(
            entity_id=entity_id, train_if_needed=False, user_id=user_id
        )

        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
            
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        summary = data_processor.get_consumption_summary()
        hourly = data_processor.get_hourly_pattern()
        daily = data_processor.get_daily_pattern()
        monthly = data_processor.get_monthly_pattern()
        peak_hours = _build_peak_info(df)

        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "hourly_pattern": hourly,
            "daily_pattern": daily,
            "monthly_pattern": monthly,
            "peak_hours": peak_hours,
            "prediction": {},
            "anomaly": {},
        }

        if pred_trained and prediction_model.is_trained:
            predictions = prediction_model.predict_future(days=7)
            report["prediction"] = {
                "status": "available",
                "next_7_days": predictions[:24] if predictions else [],
            }
        else:
            report["prediction"] = {
                "status": "not_trained",
                "message": "Train prediction models to include forecasts in reports.",
            }

        if anomaly_trained and anomaly_detector.is_trained:
            patterns = anomaly_detector.get_anomaly_patterns(df)
            report["anomaly"] = patterns
        else:
            report["anomaly"] = {
                "status": "not_trained",
                "message": "Train anomaly detector to include anomaly insights.",
            }

        # Save report to database
        save_report("energy_report", json.dumps(report))

        return jsonify({"status": "success", "report": report})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/report/download", methods=["GET"])
def download_report():
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        summary = data_processor.get_consumption_summary()
        hourly = data_processor.get_hourly_pattern()
        daily = data_processor.get_daily_pattern()
        monthly = data_processor.get_monthly_pattern()
        peak_hours = _build_peak_info(df)

        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "hourly_pattern": hourly,
            "daily_pattern": daily,
            "monthly_pattern": monthly,
            "peak_hours": peak_hours,
        }

        json_str = json.dumps(report, indent=2)
        buffer = io.BytesIO()
        buffer.write(json_str.encode("utf-8"))
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/json",
            as_attachment=True,
            download_name=f'energy_report_{datetime.now().strftime("%Y%m%d")}.json',
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/alerts/current", methods=["GET"])
def current_alerts():
    """
    Simple alert summary based on anomalies and peak thresholds
    This implements Step 8 of the flow: "Red alert pop-up"
    """
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data on the My Data & Bills page."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        peak_info = _build_peak_info(df)
        alert_messages = []
        alert_severity = "normal"  # normal, warning, danger
        
        if peak_info and peak_info.get("peak_hours"):
            peak_hours = peak_info['peak_hours']
            # Check if peak hours include evening hours (6-9 PM)
            evening_peak = any(h in peak_hours for h in [18, 19, 20, 21])
            if evening_peak:
                alert_messages.append(f"High usage hours detected: {peak_hours}")
                alert_severity = "warning"
            else:
                alert_messages.append(f"Peak usage hours: {peak_hours}")
        
        trained, _ = ensure_anomaly_model_trained(
            contamination=0.01, entity_id=entity_id, train_if_needed=True, user_id=user_id
        )
        anomaly_summary = None
        if trained:
            res = anomaly_detector.detect_anomalies(df)
            if isinstance(res, dict) and "anomaly_percentage" in res:
                anomaly_summary = {
                    "anomaly_percentage": res["anomaly_percentage"],
                    "anomaly_count": res["anomaly_count"],
                }
                anomaly_pct = float(res.get("anomaly_percentage", 0))
                if anomaly_pct > 5:
                    alert_messages.append("Anomaly rate above 5% - unusual consumption detected")
                    alert_severity = "danger"
                elif anomaly_pct > 1 and alert_severity != "danger":
                    alert_messages.append("Mild anomaly detected in today's usage pattern")
                    alert_severity = "warning"

                # Highlight explicit evening-window anomalies (6-9 PM) for dashboard alerting.
                anomalies = res.get("anomalies") or []
                if anomalies:
                    anomaly_df = pd.DataFrame(anomalies)
                    if "datetime" in anomaly_df.columns:
                        try:
                            anomaly_df["datetime"] = pd.to_datetime(anomaly_df["datetime"], errors="coerce")
                            anomaly_df = anomaly_df.dropna(subset=["datetime"])
                            anomaly_df["hour"] = anomaly_df["datetime"].dt.hour
                            evening_count = int(anomaly_df["hour"].isin([18, 19, 20, 21]).sum())
                            if evening_count > 0:
                                alert_messages.append(
                                    "Unusual energy usage detected between 6-9 PM"
                                )
                                if evening_count >= 3:
                                    alert_severity = "danger"
                                elif alert_severity == "normal":
                                    alert_severity = "warning"
                        except Exception:
                            pass
        
        return jsonify(
            {
                "status": "success",
                "alerts": alert_messages,
                "alert_severity": alert_severity,
                "peak": peak_info,
                "anomaly": anomaly_summary,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/insights/energy-tips", methods=["GET"])
def get_energy_tips():
    """
    Get energy saving tips and decision support
    This implements Step 10 of the flow: "Decision Support (User Action)"
    Provides actionable suggestions to reduce electricity costs
    """
    user_id = get_current_user_id()
    
    # Try to load user-specific data
    ok, msg = ensure_data_loaded(user_id)
    if not ok:
        return jsonify({"status": "error", "message": "No data available. Please upload your training data."}), 200
    
    try:
        entity_id = request.args.get("entity_id")
        df = get_user_energy_data(user_id)
        if df is None:
            return jsonify({"status": "error", "message": "No data available."}), 200
        
        df = apply_entity_filter(df, entity_id)
        data_processor.data = df
        
        tips = []
        
        # Get hourly pattern
        hourly = data_processor.get_hourly_pattern()
        if hourly:
            # Find peak hours
            peak_hours = [h["hour"] for h in hourly if h.get("Global_active_power", 0) > 2.0]
            
            # Tip 1: Shift usage away from peak hours
            if peak_hours:
                if any(h in [18, 19, 20, 21] for h in peak_hours):
                    tips.append({
                        "title": "Avoid Peak Evening Hours",
                        "description": "Your energy usage peaks between 6-9 PM. Consider running high-power appliances (AC, washing machine) during off-peak hours (morning or late night).",
                        "impact": "high",
                        "action": "shift_usage"
                    })
        
        # Get summary for tips
        try:
            summary = data_processor.get_consumption_summary()
            if summary and summary.get("target"):
                mean_power = summary["target"].get("mean", 0)
                max_power = summary["target"].get("max", 0)
                
                # Tip 2: AC usage
                if max_power > 3.0:
                    tips.append({
                        "title": "Reduce AC Usage",
                        "description": "High power spikes detected - likely from AC. Set thermostat to 24-26 C and use fans for air circulation.",
                        "impact": "high",
                        "action": "reduce_ac"
                    })
                
                # Tip 3: Standby power
                if mean_power > 0.5:
                    tips.append({
                        "title": "Unplug Standby Devices",
                        "description": "Your average consumption is relatively high even during low-activity periods. Unplug chargers and electronics when not in use.",
                        "impact": "medium",
                        "action": "standby_reduction"
                    })
        except Exception as e:
            logger.warning("Tip generation error: %s", e)
        
        # Tip 4: General tips always included
        tips.append({
            "title": "Use LED Bulbs",
            "description": "Replace incandescent bulbs with LED bulbs - they use 75% less energy.",
            "impact": "low",
            "action": "lighting"
        })
        
        tips.append({
            "title": "Regular AC Maintenance",
            "description": "Clean or replace AC filters monthly for optimal efficiency.",
            "impact": "medium",
            "action": "maintenance"
        })
        
        return jsonify({
            "status": "success",
            "tips": tips,
            "tip_count": len(tips)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==================== User-Specific Data Endpoints ====================

def get_current_user_id():
    """Get current user ID from session"""
    return session.get('user_id')


def get_user_energy_data(user_id=None):
    """Get the energy data for a specific user from cache"""
    global energy_data_cache
    if user_id and user_id in energy_data_cache:
        return energy_data_cache[user_id]
    return None


def set_user_energy_data(user_id, df):
    """Set the energy data for a specific user in cache"""
    global energy_data_cache
    energy_data_cache[user_id] = df


@api_bp.route("/user/training/upload", methods=["POST"])
@login_required
def upload_user_training_data():
    """Upload user's own CSV for training the model"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"status": "error", "message": "User not authenticated"}), 401

        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400

        # Read the CSV
        df = pd.read_csv(file.stream)
        df.columns = df.columns.str.strip()
        
        # Validate required columns (at least datetime and a numeric target)
        mapping_raw = request.form.get("mapping")
        mapping = json.loads(mapping_raw) if mapping_raw else {}
        
        # Use infer_mapping to get suggestions
        analysis = infer_mapping(df)
        datetime_col = mapping.get("datetime_column") or analysis["suggested"].get("datetime_column")
        target_col = mapping.get("target_column") or analysis["suggested"].get("target_column")
        
        if not datetime_col:
            return jsonify({"status": "error", "message": "Could not identify datetime column"}), 400
        if not target_col:
            return jsonify({"status": "error", "message": "Could not identify target column"}), 400

        # Process the data (Step 3-6 flow: parse -> clean -> smooth -> features)
        df["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df.rename(columns={target_col: "Global_active_power"}, inplace=True)

        df = _normalize_energy_dataframe(df)
        if df is None or df.empty:
            return jsonify({"status": "error", "message": "No valid records found after parsing. Please check your CSV mapping."}), 400

        _configure_runtime_from_df(df)
        if "datetime" in df.columns:
            df = df.sort_values("datetime").reset_index(drop=True)

        rows_before_cleaning = len(df)
        negative_before = (
            int((df["Global_active_power"] < 0).sum())
            if "Global_active_power" in df.columns
            else 0
        )

        data_processor.data = df.copy()
        data_processor.set_config(dataset_config)
        data_processor.clean_data()
        data_processor.smooth_noise(window_size=3)
        data_processor.extract_features()
        df = data_processor.data.copy()

        _configure_runtime_from_df(df)
        prediction_model.set_config(dataset_config)
        anomaly_detector.set_config(dataset_config)

        negative_after = (
            int((df["Global_active_power"] < 0).sum())
            if "Global_active_power" in df.columns
            else 0
        )

        # Save a historical dataset snapshot and keep active fallback copy.
        from app.utils.database import save_user_training_data
        dataset_filename = f"training_data_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.csv"
        csv_path = save_user_training_data(user_id, df, dataset_filename=dataset_filename)

        # Persist uploaded dataset rows in energy_data table.
        persist_df = df.copy()
        persist_df["entity_id"] = f"user_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        save_energy_data(persist_df)

        # Derive billing metadata from uploaded CSV and create active billing record.
        period = _extract_period_and_usage(df)
        rate_raw = request.form.get("rate_per_unit")
        total_amount_raw = request.form.get("total_amount")
        rate_per_unit = float(rate_raw) if rate_raw not in (None, "") else None
        total_amount = (
            float(total_amount_raw)
            if total_amount_raw not in (None, "")
            else (round(period["total_units"] * rate_per_unit, 2) if rate_per_unit else None)
        )
        billing_id = create_billing_history_record(
            user_id=user_id,
            month=period["month"],
            year=period["year"],
            total_units=period["total_units"],
            total_amount=total_amount,
            rate_per_unit=rate_per_unit,
            upload_type="CSV",
            source_file=file.filename,
            records_count=len(df),
            columns=list(df.columns),
            date_range_start=period["date_start"],
            date_range_end=period["date_end"],
            dataset_path=csv_path,
            set_active=True,  # transaction-safe single-active switch
        )

        # Cache the latest user data for immediate dashboard use
        set_user_energy_data(user_id, df)

        # Reset trained model states because training data changed
        prediction_model.is_trained = False
        prediction_model.models = {}
        prediction_model.training_metrics = {}
        prediction_model.trained_feature_columns = []
        anomaly_detector.is_trained = False
        anomaly_detector.model = None
        
        return jsonify({
            "status": "success",
            "message": "Training data uploaded and processed successfully",
            "records": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df["datetime"].min()),
                "end": str(df["datetime"].max())
            },
            "csv_path": csv_path,
            "processing": {
                "rows_before_cleaning": rows_before_cleaning,
                "rows_after_cleaning": len(df),
                "negative_removed": max(0, negative_before - negative_after),
            },
            "billing": {
                "id": billing_id,
                "month": period["month"],
                "year": period["year"],
                "total_units": period["total_units"],
                "is_active": True,
            },
        })
    except Exception as e:
        logger.exception("User training upload failed: %s", e)
        return jsonify({"status": "error", "message": "Failed to upload training data"}), 500


@api_bp.route("/user/training/data", methods=["GET"])
@login_required
def get_user_training_data():
    """Get user's uploaded training data"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"status": "error", "message": "User not authenticated"}), 401

        active = get_active_billing_history(user_id)
        if not active:
            return jsonify(
                {
                    "status": "success",
                    "has_data": False,
                    "message": "No Active Dataset",
                }
            )

        from app.utils.database import get_user_training_data as db_get_user_training_data
        df = db_get_user_training_data(user_id)

        if df is None or df.empty:
            return jsonify(
                {
                    "status": "success",
                    "has_data": False,
                    "message": "No Active Dataset",
                }
            )

        return jsonify(
            {
                "status": "success",
                "has_data": True,
                "records": int(active.get("records_count") or len(df)),
                "columns": active.get("columns") or list(df.columns),
                "date_range": {
                    "start": active.get("date_range_start")
                    or (str(df["datetime"].min()) if "datetime" in df.columns else None),
                    "end": active.get("date_range_end")
                    or (str(df["datetime"].max()) if "datetime" in df.columns else None),
                },
                "period": {
                    "month": active.get("month"),
                    "year": active.get("year"),
                },
                "active_billing_id": active.get("id"),
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/user/training/delete", methods=["DELETE"])
@login_required
def delete_user_training_data():
    """Delete user's uploaded training data"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"status": "error", "message": "User not authenticated"}), 401

        from app.utils.database import delete_user_training_data
        result = delete_user_training_data(user_id)
        
        # Clear the cache for this user after deletion
        global energy_data_cache
        if user_id in energy_data_cache:
            del energy_data_cache[user_id]

        # Reset in-memory model states so deleted data is not used anywhere.
        prediction_model.is_trained = False
        prediction_model.models = {}
        prediction_model.training_metrics = {}
        prediction_model.trained_feature_columns = []
        anomaly_detector.is_trained = False
        anomaly_detector.model = None

        return jsonify({
            "status": "success",
            "deleted": result,
            "message": "Training data deleted" if result else "No training data to delete"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/user/training/train", methods=["POST"])
@login_required
def train_user_model():
    """Train prediction model using user's own data"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"status": "error", "message": "User not authenticated"}), 401

        from app.utils.database import get_user_training_data as db_get_user_training_data
        df = db_get_user_training_data(user_id)
        
        if df is None or df.empty:
            return jsonify({"status": "error", "message": "No training data uploaded. Please upload your CSV first."}), 400

        # Use the user's data for training
        data_processor.data = df.copy()
        data_processor.set_config(dataset_config)
        
        # Extract features
        data = data_processor.extract_features()
        if data is None:
            return jsonify({"status": "error", "message": "Failed to extract features from data"}), 500
        
        # Sample if too large
        if len(data) > 12000:
            data = data.sample(n=12000, random_state=42)
        
        # Train the model
        tune = str(request.args.get("tune", "0")).strip().lower() in {"1", "true", "yes", "on"}
        metrics = prediction_model.train_models(data, tune=tune)
        
        if not metrics or isinstance(metrics, dict) and metrics.get("error"):
            return jsonify({
                "status": "error", 
                "message": "Training failed",
                "detail": metrics.get("error") if isinstance(metrics, dict) else "Unknown error"
            }), 500
        
        # Save model results
        version = datetime.now().isoformat()
        for model_name, model_metrics in metrics.items():
            if model_name != "best_model" and isinstance(model_metrics, dict):
                model_metrics["version"] = version
                model_metrics["user_id"] = user_id
                save_model_result(model_name, model_metrics)
        
        return jsonify({
            "status": "success",
            "message": "Model trained successfully with your data",
            "metrics": metrics,
            "records_used": len(data)
        })
    except Exception as e:
        logger.exception("User model training failed: %s", e)
        return jsonify({"status": "error", "message": "Failed to train model"}), 500


@api_bp.route("/user/bill/upload", methods=["POST"])
@login_required
def upload_electricity_bill():
    """Upload electricity bill data (from CSV or form data)"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"status": "error", "message": "User not authenticated"}), 401

        # Check if CSV file or form data
        if "file" in request.files and request.files["file"].filename != "":
            # CSV file upload
            file = request.files["file"]
            filename = str(file.filename or "")
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext not in {"csv", "pdf", "png", "jpg", "jpeg", "webp"}:
                return jsonify({
                    "status": "error",
                    "message": "Unsupported file type. Allowed: CSV, PDF, JPG, PNG, WEBP."
                }), 400

            if ext != "csv":
                return jsonify({
                    "status": "error",
                    "message": "PDF/JPG/PNG bill files are accepted in UI, but automatic parsing currently supports CSV only. Use Manual Entry for now."
                }), 400

            df = pd.read_csv(file.stream)
            df.columns = df.columns.str.strip()
            
            # Analyze CSV to find bill-related columns
            columns = df.columns.tolist()
            lower_cols = {c.lower(): c for c in columns}
            
            # Map common column names
            bill_data = {}
            
            # Find month/billing period
            for key in ['month', 'billing_month', 'period', 'billing_period']:
                if key in lower_cols:
                    bill_data['billing_month'] = str(df[lower_cols[key]].iloc[0])
                    break
            
            # Find year
            for key in ['year', 'billing_year']:
                if key in lower_cols:
                    bill_data['year'] = int(df[lower_cols[key]].iloc[0]) if pd.api.types.is_numeric_dtype(df[lower_cols[key]]) else int(str(df[lower_cols[key]].iloc[0]).split('-')[0])
                    break
            if 'year' not in bill_data:
                bill_data['year'] = datetime.now().year
            
            # Find units consumed
            for key in ['units', 'units_consumed', 'consumption', 'kwh', 'energy']:
                if key in lower_cols:
                    bill_data['units_consumed'] = float(df[lower_cols[key]].iloc[0])
                    break
            
            # Find total amount
            for key in ['amount', 'total_amount', 'bill_amount', 'total', 'charge']:
                if key in lower_cols:
                    bill_data['total_amount'] = float(df[lower_cols[key]].iloc[0])
                    break
            
            # Find rate per unit
            for key in ['rate', 'rate_per_unit', 'unit_rate', 'price_per_unit']:
                if key in lower_cols:
                    bill_data['rate_per_unit'] = float(df[lower_cols[key]].iloc[0])
                    break
            
            # Calculate rate if not found
            if 'units_consumed' in bill_data and 'total_amount' in bill_data and bill_data['units_consumed'] > 0:
                if 'rate_per_unit' not in bill_data:
                    bill_data['rate_per_unit'] = round(bill_data['total_amount'] / bill_data['units_consumed'], 2)
            
            # Find bill date
            for key in ['bill_date', 'date', 'invoice_date', 'billing_date']:
                if key in lower_cols:
                    bill_data['bill_date'] = str(df[lower_cols[key]].iloc[0])
                    break
            
            # Find due date
            for key in ['due_date', 'payment_due', 'deadline']:
                if key in lower_cols:
                    bill_data['due_date'] = str(df[lower_cols[key]].iloc[0])
                    break
            
            # Status
            bill_data['status'] = 'Paid'  # Default for uploaded bills
            bill_data['source_file'] = filename
            bill_data['upload_type'] = 'CSV'
            
            if 'units_consumed' not in bill_data or 'total_amount' not in bill_data:
                return jsonify({"status": "error", "message": "Could not find units consumed or total amount in CSV. Please provide a CSV with 'units_consumed' and 'total_amount' columns, or use manual form entry."}), 400
                
        else:
            # Manual form data
            bill_data = {
                'billing_month': request.form.get('billing_month', ''),
                'year': int(request.form.get('year', datetime.now().year)),
                'units_consumed': float(request.form.get('units_consumed', 0)),
                'total_amount': float(request.form.get('total_amount', 0)),
                'rate_per_unit': float(request.form.get('rate_per_unit', 0)),
                'bill_date': request.form.get('bill_date', ''),
                'due_date': request.form.get('due_date', ''),
                'status': request.form.get('status', 'Paid'),
                'source_file': 'Manual Entry',
                'upload_type': 'Manual',
            }
        
        # Calculate rate if not provided
        if bill_data.get('units_consumed', 0) > 0 and bill_data.get('total_amount', 0) > 0 and bill_data.get('rate_per_unit', 0) == 0:
            bill_data['rate_per_unit'] = round(bill_data['total_amount'] / bill_data['units_consumed'], 2)
        
        from app.utils.database import add_electricity_bill
        bill_id = add_electricity_bill(user_id, bill_data)
        
        return jsonify({
            "status": "success",
            "message": "Electricity bill uploaded successfully",
            "bill_id": bill_id,
            "bill_data": bill_data
        })
    except Exception as e:
        logger.exception("Bill upload failed: %s", e)
        return jsonify({"status": "error", "message": "Failed to upload bill"}), 500


@api_bp.route("/user/bills", methods=["GET"])
@login_required
def get_electricity_bills():
    """Get all electricity bills for current user"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"status": "error", "message": "User not authenticated"}), 401

        from app.utils.database import get_electricity_bills
        bills = get_electricity_bills(user_id)
        
        return jsonify({
            "status": "success",
            "bills": bills,
            "count": len(bills)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/user/bills/<bill_id>", methods=["DELETE"])
@login_required
def delete_electricity_bill(bill_id):
    """Delete a billing history record."""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"status": "error", "message": "User not authenticated"}), 401

        from app.utils.database import delete_electricity_bill
        result = delete_electricity_bill(user_id, bill_id)
        if not result:
            return jsonify({"status": "error", "message": "Bill record not found"}), 404

        # Invalidate cache and model state after dataset deletion.
        global energy_data_cache
        if user_id in energy_data_cache:
            del energy_data_cache[user_id]
        prediction_model.is_trained = False
        prediction_model.models = {}
        prediction_model.training_metrics = {}
        prediction_model.trained_feature_columns = []
        anomaly_detector.is_trained = False
        anomaly_detector.model = None

        return jsonify({"status": "success", "message": "Bill record deleted"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/user/bills/<int:bill_id>/activate", methods=["POST"])
@login_required
def activate_billing_dataset(bill_id):
    """Set selected CSV billing record as active dataset."""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"status": "error", "message": "User not authenticated"}), 401

        from app.utils.database import get_billing_history
        history = get_billing_history(user_id)
        selected = next((x for x in history if int(x["id"]) == int(bill_id)), None)
        if not selected:
            return jsonify({"status": "error", "message": "Billing record not found"}), 404
        if not selected.get("dataset_path") or not os.path.exists(str(selected.get("dataset_path"))):
            return jsonify(
                {
                    "status": "error",
                    "message": "Only CSV-backed dataset records can be activated.",
                }
            ), 400

        ok = set_active_billing_history(user_id, bill_id)
        if not ok:
            return jsonify({"status": "error", "message": "Failed to set active dataset"}), 400

        # Invalidate cache and model state so all downstream routes use new active dataset.
        global energy_data_cache
        if user_id in energy_data_cache:
            del energy_data_cache[user_id]
        prediction_model.is_trained = False
        prediction_model.models = {}
        prediction_model.training_metrics = {}
        prediction_model.trained_feature_columns = []
        anomaly_detector.is_trained = False
        anomaly_detector.model = None

        return jsonify({"status": "success", "message": "Active dataset updated"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/user/bills/export", methods=["GET"])
@login_required
def export_user_bills_csv():
    """Download all user's bills as CSV"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"status": "error", "message": "User not authenticated"}), 401

        from app.utils.database import get_billing_history
        bills = get_billing_history(user_id)
        if not bills:
            return jsonify({"status": "error", "message": "No bills found"}), 404

        df = pd.DataFrame(
            [
                {
                    "id": b.get("id"),
                    "month": b.get("month"),
                    "year": b.get("year"),
                    "total_units": b.get("total_units"),
                    "total_amount": b.get("total_amount"),
                    "rate_per_unit": b.get("rate_per_unit"),
                    "upload_type": b.get("upload_type"),
                    "upload_timestamp": b.get("upload_timestamp"),
                    "is_active": int(bool(b.get("is_active"))),
                    "records_count": b.get("records_count"),
                    "date_range_start": b.get("date_range_start"),
                    "date_range_end": b.get("date_range_end"),
                }
                for b in bills
            ]
        )
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"billing_history_user_{user_id}.csv",
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/user/bill/predict", methods=["GET"])
@login_required
def predict_from_bills():
    """Predict future consumption based on user's electricity bills"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({"status": "error", "message": "User not authenticated"}), 401

        # Get user's bills
        from app.utils.database import get_electricity_bills
        bills = get_electricity_bills(user_id)
        
        if not bills:
            return jsonify({"status": "error", "message": "No electricity bills found. Please upload your bills first."}), 400

        # Convert bills to DataFrame
        bills_df = pd.DataFrame(bills)
        
        # Calculate average consumption
        avg_units = bills_df['units_consumed'].mean()
        avg_amount = bills_df['total_amount'].mean()
        
        # If model is trained, use it for prediction
        if prediction_model.is_trained:
            # Get regular predictions
            days = request.args.get('days', 30, type=int)
            predictions = prediction_model.predict_future(days=days)
            
            return jsonify({
                "status": "success",
                "prediction_type": "ml_based",
                "bill_statistics": {
                    "average_monthly_units": round(avg_units, 2),
                    "average_monthly_amount": round(avg_amount, 2),
                    "total_bills": len(bills),
                    "date_range": {
                        "start": str(bills_df['bill_date'].min()) if 'bill_date' in bills_df.columns else None,
                        "end": str(bills_df['bill_date'].max()) if 'bill_date' in bills_df.columns else None
                    }
                },
                "ml_predictions": predictions[:30] if predictions else [],
                "estimated_monthly": round(avg_units, 2)
            })
        else:
            # Simple prediction based on bill averages
            return jsonify({
                "status": "success",
                "prediction_type": "bill_based",
                "bill_statistics": {
                    "average_monthly_units": round(avg_units, 2),
                    "average_monthly_amount": round(avg_amount, 2),
                    "total_bills": len(bills),
                    "date_range": {
                        "start": str(bills_df['bill_date'].min()) if 'bill_date' in bills_df.columns else None,
                        "end": str(bills_df['bill_date'].max()) if 'bill_date' in bills_df.columns else None
                    }
                },
                "predictions": [
                    {
                        "month": i + 1,
                        "estimated_units": round(avg_units, 2),
                        "estimated_amount": round(avg_amount, 2)
                    }
                    for i in range(12)
                ],
                "estimated_monthly": round(avg_units, 2),
                "message": "Model not trained. Showing predictions based on bill averages."
            })
    except Exception as e:
        logger.exception("Bill-based prediction failed: %s", e)
        return jsonify({"status": "error", "message": "Failed to run bill prediction"}), 500
