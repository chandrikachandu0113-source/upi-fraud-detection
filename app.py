
import os
import uuid
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session
)

from werkzeug.utils import secure_filename

from fraud_detector import analyze_image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

app = Flask(__name__)

# Secret key loaded from environment variable
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "dev-secret-key"
)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

ELA_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "ela"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


# ============================================================
# CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    ELA_FOLDER,
    exist_ok=True
)


# ============================================================
# TEMPORARY ANALYSIS HISTORY
# ============================================================
#
# This stores analysis results while the Flask server is running.
#
# Later, this can be replaced with MySQL/SQLite so history
# remains available even after restarting the application.
#
# ============================================================

analysis_history = []


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    # --------------------------------------------------------
    # If already logged in
    # --------------------------------------------------------

    if session.get("logged_in"):

        return redirect(
            url_for("dashboard")
        )


    # --------------------------------------------------------
    # Login form submitted
    # --------------------------------------------------------

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        # ----------------------------------------------------
        # Login credentials from environment variables
        # ----------------------------------------------------

        ADMIN_EMAIL = os.environ.get(
            "ADMIN_EMAIL"
        )

        ADMIN_PASSWORD = os.environ.get(
            "ADMIN_PASSWORD"
        )


        # ----------------------------------------------------
        # Validate login
        # ----------------------------------------------------

        if (
            email == ADMIN_EMAIL
            and
            password == ADMIN_PASSWORD
        ):

            session["logged_in"] = True

            session["user_email"] = email

            return redirect(
                url_for("dashboard")
            )


        # ----------------------------------------------------
        # Invalid login
        # ----------------------------------------------------

        return render_template(
            "login.html",
            error="Invalid email or password."
        )


    # --------------------------------------------------------
    # Show login page
    # --------------------------------------------------------

    return render_template(
        "login.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    # --------------------------------------------------------
    # Check login
    # --------------------------------------------------------

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )


    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    total_analyses = len(
        analysis_history
    )

    genuine_count = 0

    suspicious_count = 0

    high_risk_count = 0


    for item in analysis_history:

        risk_level = str(
            item.get(
                "risk_level",
                ""
            )
        ).lower()


        result = str(
            item.get(
                "result",
                ""
            )
        ).lower()


        # ----------------------------------------------------
        # Genuine
        # ----------------------------------------------------

        if (
            "genuine" in result
            or
            "low risk" in risk_level
        ):

            genuine_count += 1


        # ----------------------------------------------------
        # Suspicious
        # ----------------------------------------------------

        elif (
            "suspicious" in result
            or
            "medium" in risk_level
            or
            "medium risk" in risk_level
        ):

            suspicious_count += 1


        # ----------------------------------------------------
        # High risk
        # ----------------------------------------------------

        elif (
            "high" in risk_level
            or
            "fraud" in result
        ):

            high_risk_count += 1


    # --------------------------------------------------------
    # Send statistics to dashboard
    # --------------------------------------------------------

    return render_template(
        "dashboard.html",

        total_analyses=total_analyses,

        genuine_count=genuine_count,

        suspicious_count=suspicious_count,

        high_risk_count=high_risk_count
    )


# ============================================================
# ANALYSIS HISTORY
# ============================================================

@app.route("/history")
def history():

    # --------------------------------------------------------
    # Require login
    # --------------------------------------------------------

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )


    # --------------------------------------------------------
    # Show history
    #
    # reversed() displays newest analysis first.
    # --------------------------------------------------------

    return render_template(
        "history.html",
        history=list(
            reversed(
                analysis_history
            )
        )
    )


# ============================================================
# PROFILE
# ============================================================

@app.route("/profile")
def profile():

    # --------------------------------------------------------
    # Require login
    # --------------------------------------------------------

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )


    return render_template(
        "profile.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    # --------------------------------------------------------
    # Clear login session
    # --------------------------------------------------------

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    # --------------------------------------------------------
    # If logged in → dashboard
    # --------------------------------------------------------

    if session.get("logged_in"):

        return redirect(
            url_for("dashboard")
        )


    # --------------------------------------------------------
    # Otherwise → login
    # --------------------------------------------------------

    return redirect(
        url_for("login")
    )


# ============================================================
# UPLOAD + FRAUD ANALYSIS
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload_image():

    # --------------------------------------------------------
    # Require login
    # --------------------------------------------------------

    if not session.get("logged_in"):

        return redirect(
            url_for("login")
        )


    # --------------------------------------------------------
    # GET request
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "upload.html"
        )


    # ========================================================
    # CHECK FILE
    # ========================================================

    if "image" not in request.files:

        return render_template(
            "upload.html",
            error=(
                "No payment screenshot uploaded."
            )
        )


    file = request.files["image"]


    # --------------------------------------------------------
    # Empty filename
    # --------------------------------------------------------

    if file.filename == "":

        return render_template(
            "upload.html",
            error=(
                "Please select a payment screenshot."
            )
        )


    # --------------------------------------------------------
    # File extension validation
    # --------------------------------------------------------

    if not allowed_file(
        file.filename
    ):

        return render_template(
            "upload.html",
            error=(
                "Only PNG, JPG and JPEG "
                "images are allowed."
            )
        )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    original_filename = secure_filename(
        file.filename
    )


    unique_filename = (
        uuid.uuid4().hex
        + "_"
        + original_filename
    )


    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )


    try:

        file.save(
            file_path
        )

    except Exception as error:

        print(
            "FILE SAVE ERROR:",
            error
        )

        return render_template(
            "upload.html",
            error=(
                "Could not save the uploaded image."
            )
        )


    # ========================================================
    # ANALYZE IMAGE
    # ========================================================

    try:

        print("\n")
        print("========================================")
        print(
            "STARTING PAYMENT SCREENSHOT ANALYSIS"
        )
        print(
            "FILE:",
            unique_filename
        )
        print("========================================")
        print()


        analysis = analyze_image(
            file_path
        )


    except Exception as error:

        print(
            "ANALYSIS ERROR:",
            error
        )

        return render_template(
            "upload.html",
            error=(
                "Error analyzing image: "
                + str(error)
            )
        )


    # ========================================================
    # CHECK ANALYSIS RESULT
    # ========================================================

    if not analysis:

        return render_template(
            "upload.html",
            error=(
                "Image analysis failed."
            )
        )


    if analysis.get(
        "result"
    ) == "Error":

        return render_template(
            "upload.html",
            error=analysis.get(
                "error",
                "Unknown analysis error."
            )
        )


    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print("\n")
    print("========================================")
    print("FINAL ANALYSIS RESULT")


    print(
        "UPI APP:",
        analysis.get(
            "upi_app",
            "Unknown"
        )
    )


    print(
        "FRAUD RESULT:",
        analysis.get(
            "result",
            "Unknown"
        )
    )


    print(
        "RISK LEVEL:",
        analysis.get(
            "risk_level",
            "Unknown"
        )
    )


    print(
        "SUSPICION SCORE:",
        analysis.get(
            "suspicion_score",
            0
        )
    )


    print("========================================")
    print()


    # ========================================================
    # SAVE RESULT TO ANALYSIS HISTORY
    # ========================================================

    history_item = {

        # ----------------------------------------------------
        # Basic information
        # ----------------------------------------------------

        "id": len(analysis_history) + 1,

        "filename": unique_filename,

        "original_filename": original_filename,

        "date": datetime.now().strftime(
            "%d-%m-%Y"
        ),

        "time": datetime.now().strftime(
            "%I:%M %p"
        ),


        # ----------------------------------------------------
        # Payment information
        # ----------------------------------------------------

        "upi_app": analysis.get(
            "upi_app",
            "Unknown"
        ),

        "amount": analysis.get(
            "amount",
            "Unknown"
        ),

        "transaction_id": analysis.get(
            "transaction_id",
            "Unknown"
        ),

        "utr": analysis.get(
            "utr",
            "Unknown"
        ),

        "merchant": analysis.get(
            "merchant",
            "Unknown"
        ),

        "payment_status": analysis.get(
            "payment_status",
            "Unknown"
        ),


        # ----------------------------------------------------
        # Fraud detection information
        # ----------------------------------------------------

        "result": analysis.get(
            "result",
            "Unknown"
        ),

        "risk_level": analysis.get(
            "risk_level",
            "Unknown"
        ),

        "suspicion_score": analysis.get(
            "suspicion_score",
            0
        ),


        # ----------------------------------------------------
        # Additional analysis information
        # ----------------------------------------------------

        "ocr_quality": analysis.get(
            "ocr_quality",
            "Unknown"
        ),

        "payment_information_score": analysis.get(
            "payment_information_score",
            0
        ),

        "consistency_score": analysis.get(
            "consistency_score",
            0
        ),

        "forensic_score": analysis.get(
            "forensic_score",
            0
        ),

        "ela_score": analysis.get(
            "ela_score",
            0
        ),

        "ai_result": analysis.get(
            "ai_result",
            "Unavailable"
        ),

        "ai_confidence": analysis.get(
            "ai_confidence",
            0
        ),

        "explanation": analysis.get(
            "explanation",
            ""
        )
    }


    # --------------------------------------------------------
    # Add to history
    # --------------------------------------------------------

    analysis_history.append(
        history_item
    )


    print(
        "ANALYSIS SAVED TO HISTORY"
    )

    print(
        "HISTORY COUNT:",
        len(analysis_history)
    )

    print()


    # ========================================================
    # SHOW RESULT PAGE
    # ========================================================

    return render_template(
        "result.html",

        result=analysis,

        uploaded_image=unique_filename
    )


# ============================================================
# ERROR HANDLER - FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "upload.html",
        error=(
            "File is too large. "
            "Maximum upload size is 10 MB."
        )
    ), 413


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("========================================")
    print(
        "UPI PAYMENT SCREENSHOT "
        "FRAUD DETECTION"
    )
    print("Server starting...")
    print(
        "Open: http://127.0.0.1:5000"
    )
    print("========================================")
    print()


    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )
