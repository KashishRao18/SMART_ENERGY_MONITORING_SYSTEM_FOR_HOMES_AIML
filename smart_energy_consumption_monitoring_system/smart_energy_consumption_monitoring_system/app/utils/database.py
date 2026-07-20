"""
Database Module for Smart Energy Monitoring System
SQLite database for persistent storage
"""

import sqlite3
import pandas as pd
import os
import re
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)
DB_PATH = os.environ.get("ENERGY_DB_PATH", "app/data/energy.db")


def init_db():
    """Initialize SQLite database with required tables"""
    os.makedirs("app/data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Energy consumption data table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS energy_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            global_active_power REAL,
            global_reactive_power REAL,
            voltage REAL,
            global_intensity REAL,
            sub_metering_1 REAL,
            sub_metering_2 REAL,
            sub_metering_3 REAL,
            entity_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Index for fast queries
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_energy_datetime
        ON energy_data(datetime)
    """
    )
    # Schema migration: add entity_id if missing
    try:
        cursor.execute("ALTER TABLE energy_data ADD COLUMN entity_id TEXT")
    except sqlite3.OperationalError:
        pass

    # Model training results
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            r2_score REAL,
            rmse REAL,
            mae REAL,
            version TEXT,
            trained_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Anomaly detection results
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT,
            power REAL,
            anomaly_score REAL,
            is_anomaly INTEGER,
            detected_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Predictions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TEXT NOT NULL,
            predicted_power REAL,
            model_used TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Reports table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT,
            content TEXT,
            generated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Billing history table (CSV/manual records + active dataset state)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS billing_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            month TEXT NOT NULL,
            year INTEGER NOT NULL,
            total_units REAL NOT NULL DEFAULT 0,
            total_amount REAL,
            rate_per_unit REAL,
            upload_type TEXT NOT NULL,
            upload_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER NOT NULL DEFAULT 0,
            records_count INTEGER,
            columns_json TEXT,
            date_range_start TEXT,
            date_range_end TEXT,
            dataset_path TEXT,
            source_file TEXT
        )
    """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_billing_history_is_active
        ON billing_history(is_active)
    """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_billing_history_user_time
        ON billing_history(user_id, upload_timestamp DESC)
    """
    )
    # Enforce single active dataset per user at DB level.
    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_billing_history_one_active_per_user
        ON billing_history(user_id) WHERE is_active = 1
    """
    )

    # Dataset metadata (key/value) to persist mapping & frequency
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_meta (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Users table for authentication with email verification
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            is_verified INTEGER DEFAULT 0,
            verification_token TEXT,
            verification_otp TEXT,
            otp_expires_at TEXT,
            verified_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Add new columns to existing users table (for database migration)
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE users ADD COLUMN verification_otp TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE users ADD COLUMN otp_expires_at TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add version column to model_results if missing
    try:
        cursor.execute("ALTER TABLE model_results ADD COLUMN version TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

    logger.info("Database initialized: %s", DB_PATH)
    return DB_PATH


def save_meta(key: str, value: str):
    """Save or update a metadata key/value pair"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO dataset_meta(key, value, updated_at)
        VALUES(?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = CURRENT_TIMESTAMP
        """,
        (key, value),
    )
    conn.commit()
    conn.close()


def get_meta(key: str):
    """Get a metadata value by key"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM dataset_meta WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_all_meta():
    """Return all metadata as a dict"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT key, value FROM dataset_meta", conn)
    conn.close()
    return dict(zip(df["key"], df["value"])) if not df.empty else {}


def save_energy_data(data):
    """Save energy data to database"""
    conn = sqlite3.connect(DB_PATH)
    df = data.copy()
    df["datetime"] = df["datetime"].astype(str)
    allowed_cols = [
        "datetime",
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
        "entity_id",
    ]
    existing_cols = [c for c in allowed_cols if c in df.columns]
    df[existing_cols].to_sql("energy_data", conn, if_exists="append", index=False)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM energy_data")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_energy_data(limit=None, start_date=None, end_date=None):
    """Retrieve energy data from database"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM energy_data"
    conditions = []
    params = []

    if start_date:
        conditions.append("datetime >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("datetime <= ?")
        params.append(end_date)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY datetime"

    if limit is not None:
        try:
            safe_limit = max(1, int(limit))
            query += " LIMIT ?"
            params.append(safe_limit)
        except (TypeError, ValueError):
            pass

    df = pd.read_sql_query(query, conn, params=params if params else None)
    conn.close()
    return df


def save_model_result(model_name, metrics):
    """Save model training results"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO model_results (model_name, r2_score, rmse, mae, version) VALUES (?, ?, ?, ?, ?)",
        (
            model_name,
            metrics.get("r2_score"),
            metrics.get("rmse"),
            metrics.get("mae"),
            metrics.get("version"),
        ),
    )
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_model_results():
    """Get all model training results"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM model_results ORDER BY trained_at DESC", conn
    )
    conn.close()
    return df


def save_anomaly_result(datetime_val, power, score, is_anomaly):
    """Save anomaly detection result"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO anomaly_results (datetime, power, anomaly_score, is_anomaly) VALUES (?, ?, ?, ?)",
        (str(datetime_val), power, score, 1 if is_anomaly else 0),
    )
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_anomaly_results(anomalies_only=True):
    """Get anomaly detection results"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM anomaly_results"
    if anomalies_only:
        query += " WHERE is_anomaly = 1"
    query += " ORDER BY detected_at DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def save_prediction(datetime_val, predicted_power, model_used):
    """Save prediction result"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (datetime, predicted_power, model_used) VALUES (?, ?, ?)",
        (str(datetime_val), predicted_power, model_used),
    )
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_predictions(limit=100):
    """Get prediction results"""
    conn = sqlite3.connect(DB_PATH)
    try:
        safe_limit = max(1, int(limit))
    except (TypeError, ValueError):
        safe_limit = 100
    df = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY datetime LIMIT ?",
        conn,
        params=(safe_limit,),
    )
    conn.close()
    return df


def save_report(report_type, content):
    """Save generated report"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO reports (report_type, content) VALUES (?, ?)",
        (report_type, content),
    )
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_reports(limit=10):
    """Get saved reports"""
    conn = sqlite3.connect(DB_PATH)
    try:
        safe_limit = max(1, int(limit))
    except (TypeError, ValueError):
        safe_limit = 10
    df = pd.read_sql_query(
        "SELECT * FROM reports ORDER BY generated_at DESC LIMIT ?",
        conn,
        params=(safe_limit,),
    )
    conn.close()
    return df


def get_database_stats():
    """Get database statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    stats = {}
    tables = [
        "energy_data",
        "model_results",
        "anomaly_results",
        "predictions",
        "reports",
        "billing_history",
        "users",
    ]
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[f"{table}_count"] = cursor.fetchone()[0]
    conn.close()
    return stats


def create_user(
    username, email, password_hash, verification_token=None, role="user"
):
    """Create a new user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, verification_token, role) VALUES (?, ?, ?, ?, ?)",
            (username, email, password_hash, verification_token, role),
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id, None
    except sqlite3.IntegrityError as e:
        conn.close()
        return None, str(e)


def get_user_by_username(username):
    """Get user by username"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        (
            "SELECT id, username, email, password_hash, is_verified, "
            "created_at, role FROM users WHERE username = ?"
        ),
        (username,),
    )
    user = cursor.fetchone()
    conn.close()
    return user


def get_user_by_email(email):
    """Get user by email"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        (
            "SELECT id, username, email, password_hash, is_verified, "
            "created_at, role FROM users WHERE email = ?"
        ),
        (email,),
    )
    user = cursor.fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    """Get user by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        (
            "SELECT id, username, email, is_verified, verification_token, "
            "verification_otp, otp_expires_at, role, created_at "
            "FROM users WHERE id = ?"
        ),
        (user_id,),
    )
    user = cursor.fetchone()
    conn.close()
    return user


def verify_user(user_id):
    """Mark user as verified"""
    from datetime import datetime

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        (
            "UPDATE users "
            "SET is_verified = 1, verified_at = ?, "
            "verification_token = NULL, verification_otp = NULL, "
            "otp_expires_at = NULL "
            "WHERE id = ?"
        ),
        (datetime.now().isoformat(), user_id),
    )
    conn.commit()
    conn.close()


def update_user_otp(user_id, otp, otp_expires_at):
    """Update verification OTP and expiration for user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET verification_otp = ?, otp_expires_at = ? WHERE id = ?",
        (otp, otp_expires_at, user_id),
    )
    conn.commit()
    conn.close()


def update_user_password(user_id, new_password_hash):
    """Update user password"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (new_password_hash, user_id),
    )
    conn.commit()
    conn.close()


def get_user_role(user_id):
    """Get user role by ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


# User-specific data directories
USER_DATA_DIR = os.path.join(os.path.dirname(DB_PATH), 'user_data')


def get_user_data_dir(user_id):
    """Get or create user-specific data directory"""
    # Validate user_id to prevent path traversal attacks
    if not re.match(r'^[a-zA-Z0-9_]+$', str(user_id)):
        raise ValueError("Invalid user_id: must be alphanumeric")
    user_dir = os.path.join(USER_DATA_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def get_user_bills_csv(user_id):
    """Get path to user's electricity bills CSV file"""
    user_dir = get_user_data_dir(user_id)
    return os.path.join(user_dir, 'electricity_bills.csv')


def get_user_training_csv(user_id):
    """Get path to user's active training data CSV file."""
    active = get_active_billing_history(user_id)
    if active:
        candidate = active.get("dataset_path")
        if candidate and os.path.exists(candidate):
            return candidate
    # If user has billing history but no active row, enforce "no active dataset".
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM billing_history WHERE user_id = ?", (user_id,))
    count = cursor.fetchone()[0]
    conn.close()
    if count > 0:
        return os.path.join(get_user_data_dir(user_id), "__no_active_dataset__.csv")
    # Backward-compatible fallback for legacy users without billing history.
    user_dir = get_user_data_dir(user_id)
    return os.path.join(user_dir, 'training_data.csv')


def init_user_bills_csv(user_id):
    """Initialize empty CSV file for user's electricity bills with standard columns"""
    csv_path = get_user_bills_csv(user_id)
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=[
            'bill_id', 'billing_month', 'year', 'units_consumed', 
            'total_amount', 'rate_per_unit', 'bill_date', 
            'due_date', 'status', 'source_file', 'uploaded_at'
        ])
        df.to_csv(csv_path, index=False)
    return csv_path


def add_electricity_bill(user_id, bill_data):
    """Backward-compatible wrapper: persist bill entry into billing_history table."""
    month = str(bill_data.get("billing_month", "Unknown")).strip() or "Unknown"
    year = int(bill_data.get("year", datetime.now().year))
    total_units = float(bill_data.get("units_consumed", 0) or 0)
    total_amount = bill_data.get("total_amount")
    rate_per_unit = bill_data.get("rate_per_unit")
    return create_billing_history_record(
        user_id=user_id,
        month=month,
        year=year,
        total_units=total_units,
        total_amount=float(total_amount) if total_amount not in (None, "") else None,
        rate_per_unit=float(rate_per_unit) if rate_per_unit not in (None, "") else None,
        upload_type=str(bill_data.get("upload_type", "Manual") or "Manual"),
        source_file=str(bill_data.get("source_file", "Manual Entry") or "Manual Entry"),
        set_active=False,
    )


def get_electricity_bills(user_id):
    """Backward-compatible wrapper returning billing_history with legacy keys."""
    rows = get_billing_history(user_id)
    adapted = []
    for row in rows:
        item = dict(row)
        item["bill_id"] = str(row.get("id"))
        item["billing_month"] = row.get("month")
        item["units_consumed"] = row.get("total_units")
        item["status"] = "Active" if row.get("is_active") else "Inactive"
        dataset_path = row.get("dataset_path")
        item["file_exists"] = bool(dataset_path and os.path.exists(str(dataset_path)))
        adapted.append(item)
    return adapted


def delete_electricity_bill(user_id, bill_id):
    """Backward-compatible wrapper to delete a billing history row."""
    return delete_billing_history_record(user_id, bill_id)


def save_user_training_data(user_id, dataframe, dataset_filename=None):
    """Save a training dataset snapshot and update active fallback copy."""
    user_dir = get_user_data_dir(user_id)
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    safe_name = dataset_filename or f"training_data_{ts}.csv"
    csv_path = os.path.join(user_dir, safe_name)
    dataframe.to_csv(csv_path, index=False)
    # Keep a stable fallback file for compatibility.
    dataframe.to_csv(os.path.join(user_dir, "training_data.csv"), index=False)
    return csv_path


def get_user_training_data(user_id):
    """Get user's active training dataset."""
    csv_path = get_user_training_csv(user_id)
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def delete_user_training_data(user_id):
    """Delete the current active dataset file and deactivate active record."""
    active = get_active_billing_history(user_id)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            "UPDATE billing_history SET is_active = 0 WHERE user_id = ? AND is_active = 1",
            (user_id,),
        )
        conn.commit()
    finally:
        conn.close()

    deleted_any = False
    if active:
        active_csv = active.get("dataset_path")
        if active_csv and os.path.exists(active_csv):
            os.remove(active_csv)
            deleted_any = True

    fallback_csv = os.path.join(get_user_data_dir(user_id), "training_data.csv")
    if os.path.exists(fallback_csv):
        os.remove(fallback_csv)
        deleted_any = True
    return deleted_any


def create_billing_history_record(
    user_id,
    month,
    year,
    total_units,
    total_amount=None,
    rate_per_unit=None,
    upload_type="CSV",
    source_file=None,
    records_count=None,
    columns=None,
    date_range_start=None,
    date_range_end=None,
    dataset_path=None,
    set_active=False,
):
    """
    Create billing history row.
    If set_active=True, switches all other rows to inactive in one transaction.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN IMMEDIATE")
        if set_active:
            cursor.execute(
                "UPDATE billing_history SET is_active = 0 WHERE user_id = ?",
                (user_id,),
            )
        cursor.execute(
            """
            INSERT INTO billing_history (
                user_id, month, year, total_units, total_amount, rate_per_unit,
                upload_type, upload_timestamp, is_active, records_count, columns_json,
                date_range_start, date_range_end, dataset_path, source_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                str(month),
                int(year),
                float(total_units),
                float(total_amount) if total_amount not in (None, "") else None,
                float(rate_per_unit) if rate_per_unit not in (None, "") else None,
                str(upload_type),
                datetime.now().isoformat(),
                1 if set_active else 0,
                int(records_count) if records_count not in (None, "") else None,
                json.dumps(columns) if columns is not None else None,
                str(date_range_start) if date_range_start is not None else None,
                str(date_range_end) if date_range_end is not None else None,
                str(dataset_path) if dataset_path else None,
                str(source_file) if source_file else None,
            ),
        )
        row_id = cursor.lastrowid
        conn.commit()
        return row_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_billing_history(user_id):
    """Get all billing records for a user sorted by latest upload."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, month, year, total_units, total_amount, rate_per_unit,
               upload_type, upload_timestamp, is_active, records_count, columns_json,
               date_range_start, date_range_end, dataset_path, source_file
        FROM billing_history
        WHERE user_id = ?
        ORDER BY upload_timestamp DESC, id DESC
        """,
        (user_id,),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    for row in rows:
        raw_cols = row.get("columns_json")
        if raw_cols:
            try:
                row["columns"] = json.loads(raw_cols)
            except Exception:
                row["columns"] = []
        else:
            row["columns"] = []
        row["is_active"] = bool(row.get("is_active"))
    return rows


def get_active_billing_history(user_id):
    """Get active billing record for a user."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, month, year, total_units, total_amount, rate_per_unit,
               upload_type, upload_timestamp, is_active, records_count, columns_json,
               date_range_start, date_range_end, dataset_path, source_file
        FROM billing_history
        WHERE user_id = ? AND is_active = 1
        ORDER BY upload_timestamp DESC, id DESC
        LIMIT 1
        """,
        (user_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    raw_cols = result.get("columns_json")
    if raw_cols:
        try:
            result["columns"] = json.loads(raw_cols)
        except Exception:
            result["columns"] = []
    else:
        result["columns"] = []
    result["is_active"] = bool(result.get("is_active"))
    return result


def set_active_billing_history(user_id, billing_id):
    """Set a billing history row active and deactivate all others atomically."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            "SELECT id FROM billing_history WHERE id = ? AND user_id = ?",
            (billing_id, user_id),
        )
        if cursor.fetchone() is None:
            conn.rollback()
            return False

        cursor.execute(
            "UPDATE billing_history SET is_active = 0 WHERE user_id = ?",
            (user_id,),
        )
        cursor.execute(
            "UPDATE billing_history SET is_active = 1 WHERE id = ? AND user_id = ?",
            (billing_id, user_id),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_billing_history_record(user_id, billing_id):
    """Delete a billing history record and its dataset file if present."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    dataset_path = None
    try:
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(
            "SELECT dataset_path FROM billing_history WHERE id = ? AND user_id = ?",
            (billing_id, user_id),
        )
        row = cursor.fetchone()
        if row is None:
            conn.rollback()
            return False
        dataset_path = row[0]
        cursor.execute(
            "DELETE FROM billing_history WHERE id = ? AND user_id = ?",
            (billing_id, user_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if dataset_path and os.path.exists(str(dataset_path)):
        try:
            os.remove(str(dataset_path))
        except OSError:
            pass
    return True


if __name__ == "__main__":
    init_db()
