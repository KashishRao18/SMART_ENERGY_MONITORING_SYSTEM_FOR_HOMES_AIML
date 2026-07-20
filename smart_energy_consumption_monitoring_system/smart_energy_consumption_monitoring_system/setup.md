# Setup Guide: Move Project to Another PC Using GitHub

This guide shows how to upload this project to GitHub and run it on another PC.

## Prerequisites

- Git installed on both PCs: https://git-scm.com/downloads
- Python 3.10+ installed on both PCs
- A GitHub account

## Part 1: Push This Project to GitHub (Current PC)

1. Open terminal in project root:

   ```powershell
   cd "e:\Python Projects\smart_energy_consumption_monitoring_system"
   ```

2. Initialize Git (skip if already initialized):

   ```powershell
   git init
   ```

3. Verify `.gitignore` includes sensitive/local files:
   - `.env`
   - `*.db`
   - `flask_session/`

4. Add and commit files:

   ```powershell
   git add .
   git commit -m "Initial commit"
   ```

5. Create a new empty GitHub repository:
   - Open: https://github.com/new
   - Example repo name: `smart_energy_consumption_monitoring_system`
   - Keep it empty (do not add README/.gitignore/license from GitHub UI)

6. Add remote and push:

   ```powershell
   git branch -M main
   git remote add origin https://github.com/<your-username>/smart_energy_consumption_monitoring_system.git
   git push -u origin main
   ```

7. If prompted for password, use a GitHub Personal Access Token (PAT), not your account password.

## Part 2: Clone and Run on Another PC

1. Clone repository:

   ```powershell
   git clone https://github.com/<your-username>/smart_energy_consumption_monitoring_system.git
   cd smart_energy_consumption_monitoring_system
   ```

2. Create virtual environment:

   ```powershell
   python -m venv .venv
   ```

3. Activate virtual environment:

   ```powershell
   .\.venv\Scripts\activate
   ```

4. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

5. Create environment file:

   ```powershell
   copy .env.example .env
   ```

6. Update `.env` values as needed (`SECRET_KEY`, `FLASK_PORT`, etc.).

7. Run the app:

   ```powershell
   python run.py
   ```

8. Open in browser:
   - http://localhost:5000

## Daily Update Workflow

### On current/development PC

```powershell
git add .
git commit -m "Describe your change"
git push
```

### On other PC

```powershell
git pull
```

## Troubleshooting

- `fatal: remote origin already exists`
  - Run:
    ```powershell
    git remote remove origin
    git remote add origin https://github.com/<your-username>/smart_energy_consumption_monitoring_system.git
    ```

- `ModuleNotFoundError`
  - Ensure virtual environment is activated, then run:
    ```powershell
    pip install -r requirements.txt
    ```

- Port already in use
  - Change `FLASK_PORT` in `.env` to a free port, for example `5001`.
