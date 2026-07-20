"""
Authentication Routes - Login, Registration, OTP Verification, Logout
"""

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    current_app,
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timedelta
from typing import TypeVar, Callable, Any
import secrets

from app.utils.database import (
    create_user,
    get_user_by_username,
    get_user_by_email,
    get_user_by_id,
    get_user_role,
    verify_user,
    update_user_otp,
    update_user_password,
)
from app.utils.email_sender import send_otp_email, send_password_reset_email


F = TypeVar("F", bound=Callable[..., Any])

auth_bp = Blueprint("auth", __name__)


def login_required(f: F) -> F:
    """Decorator to require login for routes"""

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)

    return decorated_function  # type: ignore[return-value]


def admin_required(f: F) -> F:
    """Decorator to require admin role for routes"""

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth.login"))
        if session.get("role") != "admin":
            flash("Administrator access required.", "danger")
            return redirect(url_for("index"))
        return f(*args, **kwargs)

    return decorated_function  # type: ignore[return-value]


def generate_otp():
    """Generate a 6-digit OTP."""
    return f"{secrets.randbelow(1000000):06d}"


def send_and_store_otp(user_id, email, username):
    """Create OTP, store expiry, and send email."""
    otp = generate_otp()
    otp_expires_at = (datetime.now() + timedelta(minutes=10)).isoformat()
    update_user_otp(user_id, otp, otp_expires_at)
    return send_otp_email(email, username, otp)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """User login"""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Please enter both username and password.", "danger")
            return render_template("login.html")

        user = get_user_by_username(username)
        if user is None:
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

        (
            user_id,
            stored_username,
            email,
            password_hash,
            is_verified,
            created_at,
            role,
        ) = user

        if not check_password_hash(password_hash, password):
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

        if not is_verified:
            if send_and_store_otp(user_id, email, stored_username):
                flash(
                    "Your account is not verified. We sent a new OTP to your email.",
                    "warning",
                )
            else:
                flash(
                    "Your account is not verified. Unable to send OTP email right now.",
                    "danger",
                )
            return redirect(url_for("auth.verify_otp", email=email))

        session["user_id"] = user_id
        session["username"] = stored_username
        session["email"] = email
        session["role"] = role or get_user_role(user_id) or "user"
        flash(f"Welcome back, {stored_username}!", "success")
        # Redirect to home page after successful login instead of dashboard
        return redirect(url_for("index"))

    return render_template("login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """User registration"""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "danger")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        if get_user_by_username(username):
            flash("Username already exists.", "danger")
            return render_template("register.html")

        if get_user_by_email(email):
            flash("Email already registered.", "danger")
            return render_template("register.html")

        password_hash = generate_password_hash(password)
        user_id, error = create_user(
            username, email, password_hash, verification_token=None
        )

        if error:
            flash("Error creating account. Please try again.", "danger")
            return render_template("register.html")

        if send_and_store_otp(user_id, email, username):
            flash(
                "Registration successful. We sent an OTP to your email.",
                "success",
            )
        else:
            flash(
                "Registration successful, but OTP email failed. Please resend OTP.",
                "warning",
            )

        return redirect(url_for("auth.verify_otp", email=email))

    return render_template("register.html")


@auth_bp.route("/verify_otp", methods=["GET", "POST"])
def verify_otp():
    """Verify user email with OTP."""
    email = request.values.get("email", "").strip().lower()

    if request.method == "POST":
        otp = request.form.get("otp", "").strip()

        if not email or not otp:
            flash("Email and OTP are required.", "danger")
            return render_template("verify_otp.html", email=email)

        user = get_user_by_email(email)
        if not user:
            flash("User not found for this email.", "danger")
            return render_template("verify_otp.html", email=email)

        user_id = user[0]
        user_data = get_user_by_id(user_id)
        if not user_data:
            flash("User not found.", "danger")
            return render_template("verify_otp.html", email=email)

        (
            _,
            username,
            user_email,
            is_verified,
            _,
            verification_otp,
            otp_expires_at,
            _,
            role,
        ) = user_data

        if is_verified:
            session["user_id"] = user_id
            session["username"] = username
            session["email"] = user_email
            session["role"] = role or get_user_role(user_id) or "user"
            flash("Your email is already verified.", "info")
            return redirect(url_for("index"))

        if not verification_otp or not otp_expires_at:
            flash("No active OTP found. Please resend OTP.", "warning")
            return redirect(url_for("auth.resend_verification", email=email))

        expires_at = datetime.fromisoformat(otp_expires_at)
        if datetime.now() > expires_at:
            flash("OTP has expired. Please request a new one.", "warning")
            return redirect(url_for("auth.resend_verification", email=email))

        if otp != verification_otp:
            flash("Invalid OTP. Please try again.", "danger")
            return render_template("verify_otp.html", email=email)

        verify_user(user_id)
        session["user_id"] = user_id
        session["username"] = username
        session["email"] = user_email
        session["role"] = role or get_user_role(user_id) or "user"
        flash(
            "Your email has been verified successfully.",
            "success",
        )
        return redirect(url_for("index"))

    return render_template("verify_otp.html", email=email)


@auth_bp.route("/resend_verification", methods=["GET", "POST"])
def resend_verification():
    """Resend email verification OTP."""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
    else:
        email = request.args.get("email", "").strip().lower()

    if request.method == "POST":
        if not email:
            flash("Please provide your email.", "danger")
            return render_template("resend_verification.html", email=email)

        user = get_user_by_email(email)
        if not user:
            flash("No user found with this email.", "danger")
            return render_template("resend_verification.html", email=email)

        (
            user_id,
            username,
            user_email,
            _,
            is_verified,
            _,
            _,
        ) = user

        if is_verified:
            flash("Your email is already verified. Please log in.", "info")
            return redirect(url_for("auth.login"))

        if send_and_store_otp(user_id, user_email, username):
            flash("A new OTP has been sent to your email address.", "success")
            return redirect(url_for("auth.verify_otp", email=user_email))

        flash(
            "Unable to send OTP right now. Please try again later.", "danger"
        )

    return render_template("resend_verification.html", email=email)


@auth_bp.route("/logout")
def logout():
    """User logout"""
    username = session.get("username", "User")
    session.clear()
    flash(f"Goodbye, {username}! You have been logged out.", "info")
    response = redirect(url_for("auth.login"))
    # Clear session/remember cookies explicitly on logout.
    session_cookie_name = current_app.config.get("SESSION_COOKIE_NAME", "session")
    response.delete_cookie(session_cookie_name)
    response.delete_cookie("remember_token")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@auth_bp.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    """Request password reset - enter email to receive OTP"""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()

        if not email:
            flash("Please enter your email address.", "danger")
            return render_template("forgot_password.html")

        user = get_user_by_email(email)
        if not user:
            # Don't reveal that the user doesn't exist
            flash("If that email exists, we have sent a password reset OTP.", "info")
            return render_template("forgot_password.html")

        (
            user_id,
            username,
            user_email,
            _,
            _,
            _,
            _,
        ) = user

        # Generate OTP and send password reset email
        otp = generate_otp()
        otp_expires_at = (datetime.now() + timedelta(minutes=10)).isoformat()
        update_user_otp(user_id, otp, otp_expires_at)

        if send_password_reset_email(user_email, username, otp):
            flash("Password reset OTP sent to your email.", "success")
        else:
            flash("Unable to send email. Please try again later.", "danger")
            return render_template("forgot_password.html")

        return redirect(url_for("auth.reset_password", email=email))

    return render_template("forgot_password.html")


@auth_bp.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    """Verify OTP and set new password"""
    email = request.values.get("email", "").strip().lower()

    if request.method == "POST":
        otp = request.form.get("otp", "").strip()
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not email or not otp or not new_password:
            flash("All fields are required.", "danger")
            return render_template("reset_password.html", email=email)

        if len(new_password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("reset_password.html", email=email)

        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("reset_password.html", email=email)

        user = get_user_by_email(email)
        if not user:
            flash("Invalid request. User not found.", "danger")
            return redirect(url_for("auth.forgot_password"))

        user_id = user[0]
        user_data = get_user_by_id(user_id)
        if not user_data:
            flash("Invalid request. User not found.", "danger")
            return redirect(url_for("auth.forgot_password"))

        (
            _,
            username,
            user_email,
            _,
            _,
            verification_otp,
            otp_expires_at,
            _,
            _,
        ) = user_data

        if not verification_otp or not otp_expires_at:
            flash("No active OTP found. Please request a new password reset.", "warning")
            return redirect(url_for("auth.forgot_password"))

        expires_at = datetime.fromisoformat(otp_expires_at)
        if datetime.now() > expires_at:
            flash("OTP has expired. Please request a new password reset.", "warning")
            return redirect(url_for("auth.forgot_password"))

        if otp != verification_otp:
            flash("Invalid OTP. Please try again.", "danger")
            return render_template("reset_password.html", email=email)

        # Update password
        new_password_hash = generate_password_hash(new_password)
        update_user_password(user_id, new_password_hash)

        flash("Your password has been reset successfully. Please log in with your new password.", "success")
        return redirect(url_for("auth.login"))

    return render_template("reset_password.html", email=email)


@auth_bp.route("/admin/panel")
@admin_required
def admin_panel():
    """Lightweight admin panel showing database stats."""
    from app.utils.database import get_database_stats

    stats = get_database_stats()
    return render_template("admin_panel.html", stats=stats)
