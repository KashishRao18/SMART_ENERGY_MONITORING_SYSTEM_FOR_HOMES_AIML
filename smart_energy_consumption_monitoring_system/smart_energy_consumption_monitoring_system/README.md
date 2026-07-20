# Smart Energy Consumption Monitoring System

## 1. Project Overview

The **Smart Energy Consumption Monitoring System** is a Flask-based web platform for analyzing household electricity usage, tracking bills, and generating predictive insights from time-series energy data.

### Problem It Solves

Energy consumers often have fragmented consumption records and limited visibility into:
- usage patterns across time,
- billing trends,
- future consumption expectations,
- abnormal usage spikes (anomalies).

This system consolidates those workflows into one authenticated dashboard and API layer.

### Key Objectives

- Provide a secure, user-specific energy analytics workspace.
- Enable CSV-based ingestion of consumption datasets.
- Support billing history tracking and manual/CSV bill entry.
- Deliver forecasting and anomaly detection capabilities.
- Offer report-ready summaries for operational decision-making.

### Target Users

- Residential consumers tracking monthly energy costs.
- Analysts/researchers evaluating energy usage trends.
- Academic/project teams building smart-energy prototypes.
- Internal admins monitoring system metrics.

## 2. Core Features

- **User Authentication**
  - Registration, login, logout.
  - Email OTP verification for account activation.
  - Password reset via OTP.
- **Energy Data Import (CSV)**
  - Upload custom CSV training data per user.
  - Automatic datetime/target column inference with validation.
- **Active Dataset Management**
  - Single active training dataset per user (documented below).
- **Billing History Tracking**
  - Persist and retrieve user bill records.
  - Export bills as CSV.
- **Manual Bill Entry**
  - Enter bill fields directly through form/API.
  - Automatic `rate_per_unit` calculation when omitted.
- **Dashboard Analytics**
  - Time-based summaries, trend views, and usage indicators.
- **Prediction Module**
  - Train models and generate future consumption predictions.
  - Bill-based fallback predictions if ML model is not trained.
- **Reports Generation**
  - Generate and store report outputs in database tables/endpoints.
- **Anomaly Detection**
  - Train anomaly model and flag suspicious consumption patterns.

## 3. Active Dataset Logic (Important)

The current implementation enforces a **single active training dataset per user**.

- Only **one** user training dataset is active at any time.
- On new CSV upload (`/api/user/training/upload`):
  - the previous active training file is **replaced** at:
    - `app/data/user_data/<user_id>/training_data.csv`
  - the newly uploaded dataset becomes active immediately.
- Dashboard/API analytics use only this active per-user dataset.
- Historical bill records and generated DB artifacts are preserved, but **training dataset version history is not retained by default** in separate snapshots.

Implementation references:
- user training file path helpers in `app/utils/database.py`
- upload and cache refresh logic in `app/routes/api.py`

## 4. System Architecture

### Backend Framework

- **Flask 3** application with blueprint-based routing:
  - `auth_bp` for authentication and account flows
  - `api_bp` for data, analytics, prediction, and billing endpoints

### Data Layer

- **SQLite** (`ENERGY_DB_PATH`, default: `app/data/energy.db`) for:
  - users,
  - energy records,
  - model results,
  - predictions,
  - anomaly results,
  - reports,
  - metadata.
- Per-user CSV storage in `app/data/user_data/<user_id>/` for:
  - active training dataset,
  - billing history CSV.

### Runtime Modules

- `DataPreprocessor`: cleaning, smoothing, feature extraction.
- `EnergyPredictionModel`: ML training + forecast generation.
- `AnomalyDetector`: anomaly model training + anomaly scoring.

### Request Flow (High Level)

1. User authenticates and obtains session.
2. User uploads energy CSV.
3. Data is normalized, cleaned, and cached for immediate use.
4. Analytics/prediction/anomaly endpoints consume active dataset.
5. Bills are stored per user and can be exported/reused for bill-based prediction.

## 5. Folder Structure

```text
smart_energy_consumption_monitoring_system/
├── app/
│   ├── __init__.py                # Flask app factory, route registration, security headers
│   ├── config.py                  # Environment-driven configuration
│   ├── data/
│   │   ├── energy.db              # SQLite database
│   │   └── user_data/             # Per-user CSV storage (training_data.csv, electricity_bills.csv)
│   ├── models/
│   │   ├── prediction.py          # ML prediction module
│   │   └── anomaly_detection.py   # Anomaly detection module
│   ├── routes/
│   │   ├── auth.py                # Login/register/logout/OTP/password reset/admin guard
│   │   └── api.py                 # Data ingestion, analytics, reports, prediction, bills APIs
│   ├── templates/                 # Jinja2 UI templates and shared components
│   └── utils/
│       ├── database.py            # SQLite schema + data access helpers
│       ├── data_preprocessing.py  # Data preprocessing pipeline
│       └── email_sender.py        # OTP/password email sending utility
├── docs/                          # Project research documentation
├── tests/                         # Unit/integration tests (pytest)
├── requirements.txt               # Python dependencies
├── run.py                         # Application entry point
├── setup.md                       # Cross-PC GitHub setup guide
└── README.md
```

### Folder Notes

- There is no dedicated top-level `static/` or `uploads/` folder in the current codebase.
- Upload-like persistence is handled in `app/data/user_data/`.

## 6. Database Models

This project uses SQLite tables (not ORM models). Core structures:

### `users`

- `id`, `username`, `email`, `password_hash`
- `role` (default `user`)
- `is_verified`, `verification_otp`, `otp_expires_at`, `verified_at`
- `created_at`

### `energy_data`

- `id`, `datetime`
- consumption/electrical fields (`global_active_power`, `voltage`, etc.)
- `entity_id`
- `created_at`

### Billing History (CSV-backed)

- Stored per user at `app/data/user_data/<user_id>/electricity_bills.csv`
- Typical fields:
  - `bill_id`, `billing_month`, `year`
  - `units_consumed`, `total_amount`, `rate_per_unit`
  - `bill_date`, `due_date`, `status`, `source_file`, `uploaded_at`

### `BillingHistory` Model (`is_active` note)

- Current implementation uses CSV-backed bill records and does **not** persist an `is_active` column for bills.
- If you standardize billing into a DB model, recommended schema extension:
  - `id`, `user_id`, `billing_month`, `year`, `units_consumed`, `total_amount`
  - `rate_per_unit`, `bill_date`, `due_date`, `status`
  - `is_active` (BOOLEAN), `created_at`, `updated_at`

### `predictions`

- `id`, `datetime`, `predicted_power`, `model_used`, `created_at`

### Additional Operational Tables

- `model_results` (training metrics)
- `anomaly_results`
- `reports`
- `dataset_meta` (configuration/mapping metadata)

## 7. Installation Guide

### Prerequisites

- Python 3.10+
- Git
- pip

### Steps

1. Clone repository:

```bash
git clone https://github.com/<your-username>/smart_energy_consumption_monitoring_system.git
cd smart_energy_consumption_monitoring_system
```

2. Create virtual environment:

```bash
python -m venv .venv
```

3. Activate virtual environment:

```bash
# Windows
.\.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Configure environment:

```bash
# Linux/macOS
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

6. Run database initialization/migrations:

```bash
python run.py
```

Notes:
- This project does not use Alembic/Flask-Migrate.
- Schema creation/migration is handled programmatically via `init_db()` during app startup.

7. Start server:

```bash
python run.py
```

8. Access application:

- `http://localhost:5000`

### Production Start (WSGI)

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

## 8. Environment Variables

Configure in `.env`:

- `SECRET_KEY`: Flask secret key for session security.
- `FLASK_DEBUG`: `true`/`false`.
- `FLASK_HOST`: host bind address (default `0.0.0.0`).
- `FLASK_PORT`: server port (default `5000`).
- `ENERGY_DB_PATH`: SQLite database path.
- `MAX_CONTENT_LENGTH_MB`: request body/upload size limit.
- `SESSION_TYPE`: session storage backend (default filesystem).
- `SESSION_USE_SIGNER`: session signing toggle.
- `SESSION_COOKIE_SECURE`: secure cookie flag (enable in HTTPS production).
- `SESSION_FILE_DIR`: server-side session file directory.
- `EMAIL_SENDER_EMAIL`: sender address for OTP/reset emails.
- `EMAIL_SENDER_PASSWORD`: app password/token for SMTP.
- `EMAIL_SMTP_SERVER`: SMTP host.
- `EMAIL_SMTP_PORT`: SMTP port.
- `EMAIL_FALLBACK_CONSOLE`: log OTP to console when email credentials are absent.

## 9. Usage Guide

### Login

1. Register account.
2. Verify account using email OTP.
3. Log in at `/auth/login`.

### Upload Energy CSV

1. Navigate to **My Data & Bills** page.
2. Upload CSV through training upload form.
3. System parses, cleans, and stores active training data.

### Active Dataset Behavior

1. Uploading a new training CSV replaces prior active per-user training file.
2. Subsequent analytics/prediction calls use this active dataset.
3. If training data is deleted, cached data/model states are reset.

### Billing History

1. Upload bill CSV or use manual form entry.
2. View all bills in billing history section.
3. Delete individual bills if needed.
4. Export billing history CSV from API/UI.

### Generate Reports

1. Open Reports page.
2. Trigger report-oriented analytics endpoints.
3. Review generated summaries/insights.

## 10. Security Features

- **Route Protection**
  - `login_required` and `admin_required` decorators for protected pages.
  - API-level auth enforcement for non-public endpoints.
- **Session Management**
  - Server-side sessions via `Flask-Session` (filesystem).
  - Explicit session and cookie clearing on logout.
- **Cache Control After Logout**
  - `Cache-Control`, `Pragma`, and `Expires` headers set to prevent back-button exposure of protected content.
- **Input Validation**
  - Required-field checks for auth and upload flows.
  - CSV presence/type checks for uploads.
  - Data normalization and null/invalid filtering in preprocessing.
  - Path traversal-safe user directory handling via user ID validation.

## 11. Tech Stack

- **Language**: Python 3
- **Backend Framework**: Flask
- **Session Layer**: Flask-Session
- **Database**: SQLite
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Production Server**: Gunicorn
- **Frontend**: Jinja2 templates + Bootstrap + Plotly.js
- **Testing**: pytest

## 12. Future Improvements

- Dataset versioning with explicit `is_active` metadata and rollback support.
- Multi-user dataset isolation hardening and tenancy controls.
- Full role-based access control (RBAC) expansion.
- REST API documentation (OpenAPI/Swagger) and token-based auth for integrations.
- Cloud deployment profiles (Docker/Kubernetes, managed DB, object storage).
- Async task processing for model training/report generation.
- Automated CI/CD with linting, tests, and release tagging.

## 13. License

This project currently has **no license file committed**.  
For open-source distribution, add a license (for example, MIT/Apache-2.0) as `LICENSE`.

## 14. Author

- **Project Team / Maintainer**: Smart Energy Consumption Monitoring System contributors
- Update this section with your name, organization, and contact details for production handover.
