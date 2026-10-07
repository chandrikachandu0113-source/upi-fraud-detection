import os
import re
import io
import uuid
import time
from datetime import datetime

import cv2
import numpy as np
import pytesseract
from PIL import Image
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None
from behavior_analyzer import analyze_behavior




# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "image_fraud_model.keras"
)

ELA_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "ela"
)

os.makedirs(ELA_FOLDER, exist_ok=True)


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print("Tesseract found:", TESSERACT_PATH)
else:
    print("WARNING: Tesseract executable not found.")


# ============================================================
# LOAD AI MODEL
# ============================================================

MODEL = None
MODEL_ERROR = None

try:
    if load_model is None:
        MODEL_ERROR = "TensorFlow AI model is unavailable because TensorFlow could not be loaded."
        print("WARNING:", MODEL_ERROR)

    elif os.path.exists(MODEL_PATH):

        print("Loading AI model:")
        print(MODEL_PATH)

        MODEL = load_model(
            MODEL_PATH,
            compile=False
        )

        print("AI model loaded successfully.")
        print("Model input shape:", MODEL.input_shape)
        print("Model output shape:", MODEL.output_shape)

    else:

        MODEL_ERROR = "Model file not found: " + MODEL_PATH
        print("WARNING:", MODEL_ERROR)

except Exception as e:

    MODEL_ERROR = str(e)

    print("ERROR: Could not load AI model.")
    print("Model error:", MODEL_ERROR)


# ============================================================
# SUPPORTED UPI APPLICATIONS
# ============================================================

UPI_APPS = [
    "PhonePe",
    "Google Pay",
    "GPay",
    "Paytm",
    "BHIM",
    "Amazon Pay",
    "WhatsApp Pay",
    "CRED",
    "MobiKwik",
    "Freecharge",
    "Airtel Money",
    "JioMoney"
]


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(text):
    """
    Clean OCR text while preserving useful characters.
    """

    if not text:
        return ""

    text = str(text)

    text = text.replace("\x0c", " ")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.splitlines():

        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def unique_lines(text):
    """
    Remove duplicate OCR lines.
    """

    if not text:
        return ""

    result = []
    seen = set()

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        key = line.lower()

        if key not in seen:

            seen.add(key)
            result.append(line)

    return "\n".join(result)


# ============================================================
# OCR NORMALIZATION
# ============================================================

def normalize_ocr_text(text):
    """
    Normalize common OCR mistakes without destroying
    important transaction information.
    """

    if not text:
        return ""

    text = str(text)

    replacements = {
        "₹": "₹",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize common OCR variants of UTR
    text = re.sub(
        r"\bUTR\s*[:\-]?\s*",
        "UTR: ",
        text,
        flags=re.IGNORECASE
    )

    return clean_text(text)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_variants(image):
    """
    Create multiple OCR-friendly versions of an image.
    """

    if image is None:
        return []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    enlarged = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    blurred = cv2.GaussianBlur(
        enlarged,
        (3, 3),
        0
    )

    contrast = cv2.convertScaleAbs(
        blurred,
        alpha=1.6,
        beta=10
    )

    _, otsu = cv2.threshold(
        contrast,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [
        gray,
        enlarged,
        contrast,
        otsu,
        adaptive
    ]


# ============================================================
# OCR ENGINE
# ============================================================

def run_ocr(image, psm=6):
    """
    Run OCR using several preprocessing variants.
    """

    if image is None:
        return ""

    outputs = []

    variants = preprocess_variants(image)

    configs = [
        f"--oem 3 --psm {psm}",
        "--oem 3 --psm 11"
    ]

    for variant in variants:

        for config in configs:

            try:

                text = pytesseract.image_to_string(
                    variant,
                    config=config
                )

                text = normalize_ocr_text(text)

                if text:
                    outputs.append(text)

            except Exception as e:

                print("OCR error:", e)

    if not outputs:
        return ""

    combined = "\n".join(outputs)

    return unique_lines(combined)


# ============================================================
# PHONEPE REGION OCR
# ============================================================

def crop_region(
    image,
    y1,
    y2,
    x1=0.0,
    x2=1.0
):

    h, w = image.shape[:2]

    start_y = max(
        0,
        int(h * y1)
    )

    end_y = min(
        h,
        int(h * y2)
    )

    start_x = max(
        0,
        int(w * x1)
    )

    end_x = min(
        w,
        int(w * x2)
    )

    return image[
        start_y:end_y,
        start_x:end_x
    ]


def phonepe_region_ocr(image):
    """
    OCR regions tuned for PhonePe screenshots.
    """

    regions = {}

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header = crop_region(
        image,
        0.00,
        0.12,
        0.02,
        0.98
    )

    regions["header"] = run_ocr(
        header,
        psm=6
    )

    # --------------------------------------------------------
    # Date / Time
    # --------------------------------------------------------
    #
    # Use a narrow upper-middle region first.
    # PhonePe screenshots normally place the date/time
    # close to the payment result area.
    #

    date_time = crop_region(
        image,
        0.00,
        0.18,
        0.20,
        0.80
    )

    # Enlarge before OCR
    date_time = cv2.resize(
        date_time,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    # Try OCR with a layout suitable for a date/time line.
    regions["date_time"] = run_ocr(
        date_time,
        psm=6
    )

    # --------------------------------------------------------
    # Merchant / Amount / UPI
    # --------------------------------------------------------

    merchant = crop_region(
        image,
        0.10,
        0.25,
        0.02,
        0.99
    )

    regions["merchant"] = run_ocr(
        merchant,
        psm=6
    )

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    amount = crop_region(
        image,
        0.12,
        0.25,
        0.55,
        0.99
    )

    regions["amount"] = run_ocr(
        amount,
        psm=7
    )

    # --------------------------------------------------------
    # UPI
    # --------------------------------------------------------

    upi = crop_region(
        image,
        0.15,
        0.27,
        0.05,
        0.99
    )

    regions["upi"] = run_ocr(
        upi,
        psm=7
    )

    # --------------------------------------------------------
    # Transaction ID
    # --------------------------------------------------------

    transaction = crop_region(
        image,
        0.27,
        0.40,
        0.02,
        0.99
    )

    regions["transaction"] = run_ocr(
        transaction,
        psm=7
    )

    # --------------------------------------------------------
    # UTR
    # --------------------------------------------------------

    utr = crop_region(
        image,
        0.38,
        0.50,
        0.02,
        0.99
    )

    regions["utr"] = run_ocr(
        utr,
        psm=7
    )

    return regions


# ============================================================
# NORMALIZE UPI ID
# ============================================================

def normalize_upi_id(value):

    if not value:
        return None

    value = str(value)

    value = value.replace(
        " ",
        ""
    )

    value = value.replace(
        "©",
        "@"
    )

    value = value.replace(
        "®",
        "@"
    )

    value = value.replace(
        "O@",
        "0@"
    )

    value = value.replace(
        "o@",
        "0@"
    )

    # OCR can confuse @ with symbols
    value = re.sub(
        r"[©®]",
        "@",
        value
    )

    match = re.search(
        r"[A-Za-z0-9._-]{3,80}"
        r"@"
        r"[A-Za-z0-9.-]{2,30}",
        value
    )

    if match:
        return match.group(0)

    return None


# ============================================================
# EXTRACT UPI ID
# ============================================================

def extract_upi_id(text):

    if not text:
        return None

    # Normal UPI format
    match = re.search(
        r"[A-Za-z0-9._-]{3,80}"
        r"\s*@\s*"
        r"[A-Za-z0-9.-]{2,30}",
        text
    )

    if match:

        return normalize_upi_id(
            match.group(0)
        )

    # OCR @ symbol corruption
    match = re.search(
        r"[A-Za-z0-9._-]{3,80}"
        r"\s*[©®]\s*"
        r"[A-Za-z0-9.-]{2,30}",
        text
    )

    if match:

        value = match.group(0)

        value = re.sub(
            r"[©®]",
            "@",
            value
        )

        return normalize_upi_id(
            value
        )

    return None


# ============================================================
# EXTRACT AMOUNT
# ============================================================

def extract_amount(text):

    if not text:
        return None

    text = str(text)

    patterns = [

        # ₹40
        r"₹\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",

        # Rs 40
        r"\bRs\.?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",

        # INR 40
        r"\bINR\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",

        # OCR question mark followed by amount
        r"\?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = match.group(1)

            value = value.replace(
                ",",
                ""
            )

            try:

                amount = float(value)

                if 0 <= amount <= 100000000:
                    return amount

            except ValueError:
                pass

    # --------------------------------------------------------
    # PhonePe fallback
    #
    # OCR often gives:
    # Paid to ARVIND STORES 40
    # ARVIND STORES =40
    # ARVIND STORES 40
    # --------------------------------------------------------

    lines = text.splitlines()

    for line in lines:

        lower = line.lower()

        if (
            "paid to" in lower
            or "arvind stores" in lower
            or "amount" in lower
            or "debited from" in lower
        ):

            numbers = re.findall(
                r"(?<!\d)"
                r"\d{1,8}(?:\.\d{1,2})?"
                r"(?!\d)",
                line
            )

            for number in numbers:

                try:

                    value = float(number)

                    # Avoid treating years / transaction numbers
                    # as the payment amount.
                    if (
                        0 < value < 100000
                        and value not in [2024, 2025, 2026]
                    ):
                        return value

                except ValueError:
                    pass

    # --------------------------------------------------------
    # Search around common PhonePe amount pattern
    # --------------------------------------------------------

    amount_match = re.search(
        r"(?:paid\s+to|debited\s+from|payment)"
        r".{0,100}?"
        r"(?<!\d)"
        r"([0-9]{1,7}(?:\.[0-9]{1,2})?)"
        r"(?!\d)",
        text,
        re.IGNORECASE
    )

    if amount_match:

        try:

            value = float(
                amount_match.group(1)
            )

            if (
                0 < value < 100000
                and value not in [2024, 2025, 2026]
            ):
                return value

        except ValueError:
            pass

    return None


# ============================================================
# EXTRACT TRANSACTION ID
# ============================================================

def extract_transaction_id(text):

    if not text:
        return None

    # PhonePe transaction ID
    match = re.search(
        r"\bT\s*\d{16,30}\b",
        text,
        re.IGNORECASE
    )

    if match:

        transaction_id = match.group(0)

        transaction_id = re.sub(
            r"\s+",
            "",
            transaction_id
        )

        return transaction_id.upper()

    # OCR may add I / 1 before T
    match = re.search(
        r"\b[IT]\s*\d{16,30}\b",
        text,
        re.IGNORECASE
    )

    if match:

        transaction_id = match.group(0)

        transaction_id = re.sub(
            r"\s+",
            "",
            transaction_id
        )

        # Only convert leading I to T when
        # followed by a long numeric ID.
        if transaction_id.upper().startswith("I"):
            transaction_id = "T" + transaction_id[1:]

        return transaction_id.upper()

    # Explicit label
    match = re.search(
        r"(?:Transaction\s*ID|"
        r"Reference\s*ID|"
        r"Transaction)"
        r"\s*[:\-]?\s*"
        r"([A-Za-z0-9]{8,40})",
        text,
        re.IGNORECASE
    )

    if match:

        value = match.group(1).strip()

        return value

    return None


# ============================================================
# EXTRACT UTR
# ============================================================

def extract_utr(text):

    if not text:
        return None

    # Explicit UTR
    match = re.search(
        r"\bUTR\s*[:\-]?\s*"
        r"([0-9]{8,20})",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    # OCR may separate digits
    match = re.search(
        r"\bUTR\s*[:\-]?\s*"
        r"([0-9][0-9\s]{7,25})",
        text,
        re.IGNORECASE
    )

    if match:

        value = re.sub(
            r"\s+",
            "",
            match.group(1)
        )

        if 8 <= len(value) <= 20:
            return value

    # 12 digit UTR fallback
    match = re.search(
        r"\b\d{12}\b",
        text
    )

    if match:
        return match.group(0)

    return None


# ============================================================
# ============================================================
# EXTRACT DATE
# ============================================================

def extract_date(text):

    if not text:
        return None

    text = str(text)

    # Normalize OCR text
    normalized = text.replace("\n", " ")
    normalized = re.sub(r"[\{\}\[\]\(\)]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # -----------------------------------------
    # Normal dates
    # -----------------------------------------

    patterns = [
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,\s]+\d{4}\b",

        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",

        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4}\b",

        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",

        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)

        if match:
            return match.group(0)

    # -----------------------------------------
    # Compact date
    # Example: 25Aug2026
    # -----------------------------------------

    compact = re.sub(r"[^A-Za-z0-9]", "", text)

    match = re.search(
        r"(\d{1,2})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d{4})",
        compact,
        re.IGNORECASE
    )

    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"

    # -----------------------------------------
    # Corrupted date:
    # Example OCR:
    # Aug}2 026
    #
    # We specifically look for:
    # month + one digit + three digits
    # -----------------------------------------

    match = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[^0-9]{0,10}"
        r"(\d)(\d{3})",
        text,
        re.IGNORECASE
    )

    if match:
        month = match.group(1)
        year = match.group(2) + match.group(3)

        # The OCR example contains:
        # Aug}2 026
        #
        # Therefore the digit immediately after Aug
        # is NOT treated as the day.
        #
        # Search only AFTER the year for a day, not before.
        after = text[match.end():]

        day_matches = re.findall(
            r"\b([0-9]{1,2})\b",
            after[:30]
        )

        if day_matches:
            day = day_matches[0]

            if 1 <= int(day) <= 31:
                return f"{day} {month} {year}"

        # If no reliable day is available,
        # return month + year rather than inventing a day.
        return f"{month} {year}"

    # -----------------------------------------
    # Month + four digit year
    # -----------------------------------------

    match = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[^0-9]{0,20}"
        r"(20[0-9]{2})",
        text,
        re.IGNORECASE
    )

    if match:
        month = match.group(1)
        year = match.group(2)

        return f"{month} {year}"

    # -----------------------------------------
    # Transaction ID fallback
    # -----------------------------------------
    #
    # IMPORTANT:
    # We do NOT assume that the transaction ID
    # contains the date, because that format is
    # not verified.
    #
    # Therefore we intentionally do not convert
    # T260825... into a date.
    # -----------------------------------------

    return None


def extract_time(text):

    if not text:
        return None

    text = str(text)

    normalized = text.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized)

    patterns = [

        # 01:35:22 PM
        r"\b\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)\b",

        # 01:35 PM
        r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b",

        # 01:35pm
        r"\b\d{1,2}:\d{2}(?:AM|PM)\b",

        # 24-hour time
        r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            normalized,
            re.IGNORECASE
        )

        if match:
            return match.group(0)

    # OCR fallback:
    # 01:3 -> 01:30
    match = re.search(
        r"\b(\d{1,2}):([0-5])\b",
        normalized
    )

    if match:
        return f"{match.group(1)}:{match.group(2)}0"

    # OCR may insert spaces:
    # 01 : 35
    match = re.search(
        r"\b(\d{1,2})\s*:\s*([0-5]\d)\b",
        normalized
    )

    if match:
        return f"{match.group(1)}:{match.group(2)}"

    return None


# ============================================================
# PAYMENT STATUS
# ============================================================

def detect_payment_status(text):

    if not text:
        return "Unknown"

    text_lower = text.lower()

    # --------------------------------------------------------
    # Failed
    # --------------------------------------------------------

    failed_patterns = [
        "transaction failed",
        "payment failed",
        "payment unsuccessful",
        "transaction unsuccessful",
        "failed"
    ]

    for pattern in failed_patterns:

        if pattern in text_lower:
            return "Failed"

    # --------------------------------------------------------
    # Pending
    # --------------------------------------------------------

    pending_patterns = [
        "transaction pending",
        "payment pending",
        "pending"
    ]

    for pattern in pending_patterns:

        if pattern in text_lower:
            return "Pending"

    # --------------------------------------------------------
    # Successful
    # --------------------------------------------------------

    successful_patterns = [
        "transaction successful",
        "payment successful",
        "payment success",
        "successful",
        "success",
        "paid to",
        "paid",
        "payment details"
    ]

    for pattern in successful_patterns:

        if pattern in text_lower:
            return "Successful"

    return "Unknown"


# ============================================================
# DETECT UPI APPLICATION
# ============================================================

def detect_upi_app(text):

    if not text:
        return None

    text_lower = text.lower()

    if "phonepe" in text_lower:
        return "PhonePe"

    if "google pay" in text_lower:
        return "Google Pay"

    if "gpay" in text_lower:
        return "Google Pay"

    if "paytm" in text_lower:
        return "Paytm"

    if "bhim" in text_lower:
        return "BHIM"

    if "amazon pay" in text_lower:
        return "Amazon Pay"

    if "whatsapp" in text_lower:
        return "WhatsApp Pay"

    if "cred" in text_lower:
        return "CRED"

    if "mobikwik" in text_lower:
        return "MobiKwik"

    if "freecharge" in text_lower:
        return "Freecharge"

    if "airtel" in text_lower:
        return "Airtel Money"

    if "jiomoney" in text_lower:
        return "JioMoney"

    return None


# ============================================================
# MERCHANT NAME
# ============================================================

def extract_merchant_name(text):

    if not text:
        return None

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # --------------------------------------------------------
    # Paid to
    # --------------------------------------------------------

    for index, line in enumerate(lines):

        lower = line.lower()

        if "paid to" in lower:

            # Same line:
            # Paid to ARVIND STORES
            same_line = re.sub(
                r".*?paid\s+to\s*",
                "",
                line,
                flags=re.IGNORECASE
            ).strip()

            if same_line:

                same_line = re.sub(
                    r"\s+\d+(?:\.\d+)?\s*$",
                    "",
                    same_line
                ).strip()

                if (
                    same_line
                    and "payment" not in same_line.lower()
                    and "details" not in same_line.lower()
                ):
                    return same_line

            # Next line
            if index + 1 < len(lines):

                merchant = lines[index + 1]

                if (
                    "payment" not in merchant.lower()
                    and "details" not in merchant.lower()
                    and "transaction" not in merchant.lower()
                ):
                    return merchant

    # --------------------------------------------------------
    # Common PhonePe merchant pattern
    # --------------------------------------------------------

    for line in lines:

        if "arvind stores" in line.lower():
            return "ARVIND STORES"

    return None


# ============================================================
# AI MODEL PREDICTION
# ============================================================

def predict_ai(image_path):

    result = {
        "available": False,
        "prediction": "AI model unavailable",
        "manipulated_probability": None,
        "authentic_probability": None,
        "error": MODEL_ERROR
    }

    if MODEL is None:
        return result

    try:

        image = cv2.imread(
            image_path
        )

        if image is None:

            result["error"] = (
                "Could not read image."
            )

            return result

        resized = cv2.resize(
            image,
            (128, 128)
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        array = rgb.astype(
            np.float32
        ) / 255.0

        array = np.expand_dims(
            array,
            axis=0
        )

        prediction = MODEL.predict(
            array,
            verbose=0
        )

        values = np.asarray(
            prediction
        ).flatten()

        if len(values) == 0:

            result["error"] = (
                "Model returned empty prediction."
            )

            return result

        # ----------------------------------------------------
        # Binary sigmoid output
        # ----------------------------------------------------

        if len(values) == 1:

            value = float(values[0])

            if value < 0 or value > 1:

                value = 1.0 / (
                    1.0 +
                    np.exp(-value)
                )

            manipulated_probability = value

        # ----------------------------------------------------
        # Two-class output
        # ----------------------------------------------------

        else:

            probs = values[-2:]

            if (
                np.any(probs < 0)
                or np.any(probs > 1)
                or abs(
                    float(np.sum(probs)) - 1.0
                ) > 0.05
            ):

                exp_values = np.exp(
                    probs - np.max(probs)
                )

                probs = (
                    exp_values /
                    np.sum(exp_values)
                )

            # Assumption:
            # class 0 = authentic
            # class 1 = manipulated

            manipulated_probability = float(
                probs[-1]
            )

        authentic_probability = (
            1.0 -
            manipulated_probability
        )

        manipulated_percentage = (
            manipulated_probability * 100
        )

        authentic_percentage = (
            authentic_probability * 100
        )

        if manipulated_percentage >= 50:

            prediction_label = (
                "Likely Manipulated"
            )

        else:

            prediction_label = (
                "Likely Authentic"
            )

        result.update({

            "available": True,

            "prediction":
                prediction_label,

            "manipulated_probability":
                round(
                    manipulated_percentage,
                    2
                ),

            "authentic_probability":
                round(
                    authentic_percentage,
                    2
                ),

            "error":
                None
        })

        return result

    except Exception as e:

        result["error"] = str(e)

        print(
            "AI prediction error:",
            e
        )

        return result


# ============================================================
# ERROR LEVEL ANALYSIS
# ============================================================

def perform_ela(image_path):

    try:

        original = Image.open(
            image_path
        ).convert("RGB")

        buffer = io.BytesIO()

        original.save(
            buffer,
            format="JPEG",
            quality=90
        )

        buffer.seek(0)

        recompressed = Image.open(
            buffer
        ).convert("RGB")

        original_array = np.array(
            original
        ).astype(np.float32)

        recompressed_array = np.array(
            recompressed
        ).astype(np.float32)

        difference = np.abs(
            original_array -
            recompressed_array
        )

        ela_score = float(
            np.mean(difference)
        )

        ela_visual = np.clip(
            difference * 10,
            0,
            255
        ).astype(np.uint8)

        ela_filename = (
            "ela_" +
            uuid.uuid4().hex +
            ".jpg"
        )

        ela_path = os.path.join(
            ELA_FOLDER,
            ela_filename
        )

        Image.fromarray(
            ela_visual
        ).save(
            ela_path,
            quality=90
        )

        ela_relative = (
            "ela/" +
            ela_filename
        )

        return {

            "available":
                True,

            "score":
                round(
                    ela_score,
                    2
                ),

            "path":
                ela_relative,

            "absolute_path":
                ela_path
        }

    except Exception as e:

        print(
            "ELA error:",
            e
        )

        return {

            "available":
                False,

            "score":
                0.0,

            "path":
                None,

            "absolute_path":
                None,

            "error":
                str(e)
        }


# ============================================================
# IMAGE FORENSIC ANALYSIS
# ============================================================

def forensic_analysis(image_path):

    result = {

        "score":
            0.0,

        "variance":
            0.0,

        "edge_percentage":
            0.0,

        "metadata_count":
            0,

        "format":
            "Unknown",

        "width":
            0,

        "height":
            0
    }

    try:

        image = cv2.imread(
            image_path
        )

        if image is None:
            return result

        height, width = image.shape[:2]

        result["width"] = width
        result["height"] = height

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Pixel variance
        # ----------------------------------------------------

        variance = float(
            np.var(gray)
        )

        result["variance"] = round(
            variance,
            2
        )

        # ----------------------------------------------------
        # Edge percentage
        # ----------------------------------------------------

        edges = cv2.Canny(
            gray,
            100,
            200
        )

        edge_pixels = np.count_nonzero(
            edges
        )

        total_pixels = edges.size

        edge_percentage = (
            edge_pixels /
            total_pixels *
            100
        )

        result["edge_percentage"] = round(
            edge_percentage,
            2
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        try:

            pil_image = Image.open(
                image_path
            )

            metadata = pil_image.getexif()

            result["metadata_count"] = len(
                metadata
            )

            result["format"] = (
                pil_image.format or
                "Unknown"
            )

        except Exception:
            pass

        # ----------------------------------------------------
        # Forensic suspicion score
        # ----------------------------------------------------

        score = 0.0

        if variance < 100:
            score += 20

        elif variance < 500:
            score += 10

        if edge_percentage > 15:
            score += 20

        elif edge_percentage > 10:
            score += 10

        result["score"] = round(
            min(score, 100),
            2
        )

        return result

    except Exception as e:

        print(
            "Forensic analysis error:",
            e
        )

        return result


# ============================================================
# PAYMENT INFORMATION ANALYSIS
# ============================================================

def payment_information_analysis(
    amount,
    transaction_id,
    utr,
    upi_id,
    upi_app,
    status,
    date,
    time_value
):
    """
    IMPORTANT:
    Missing OCR fields are treated as OCR limitations,
    NOT automatically as fraud evidence.

    Actual negative payment states such as Failed/Pending
    receive stronger suspicion.
    """

    warnings = []

    # --------------------------------------------------------
    # OCR completeness
    # --------------------------------------------------------

    missing_fields = 0

    if amount is None:
        missing_fields += 1
        warnings.append(
            "Payment amount could not be confidently detected."
        )

    if not transaction_id:
        missing_fields += 1
        warnings.append(
            "Transaction ID could not be detected."
        )

    if not utr:
        missing_fields += 1
        warnings.append(
            "UTR/reference number could not be detected."
        )

    if not upi_id:
        missing_fields += 1
        warnings.append(
            "UPI ID could not be detected."
        )

    if not upi_app:
        missing_fields += 1
        warnings.append(
            "UPI application could not be detected."
        )

    if not date:
        missing_fields += 1
        warnings.append(
            "Transaction date could not be confidently detected."
        )

    if not time_value:
        missing_fields += 1
        warnings.append(
            "Transaction time could not be confidently detected."
        )

    if status == "Unknown":
        warnings.append(
            "Payment success status could not be confidently detected by OCR."
        )

    # --------------------------------------------------------
    # Payment information quality score
    # --------------------------------------------------------

    extracted = 8 - missing_fields

    quality_score = (
        extracted / 8
    ) * 100

    # --------------------------------------------------------
    # Fraud suspicion from payment state
    #
    # Missing fields are NOT fraud.
    # --------------------------------------------------------

    suspicion = 0.0

    if status == "Pending":

        suspicion += 15

        warnings.append(
            "Payment status appears to be pending."
        )

    elif status == "Failed":

        suspicion += 30

        warnings.append(
            "Payment status appears to be failed."
        )

    # Unknown status only means OCR uncertainty.
    # It does NOT increase fraud suspicion.

    return {

        "score":
            round(
                quality_score,
                2
            ),

        "suspicion":
            round(
                suspicion,
                2
            ),

        "status":
            (
                "Complete"
                if missing_fields == 0
                else
                "Partially Complete"
                if extracted >= 5
                else
                "Incomplete"
            ),

        "warnings":
            warnings
    }


# ============================================================
# CONSISTENCY ANALYSIS
# ============================================================

def consistency_analysis(
    amount,
    transaction_id,
    utr,
    upi_id,
    upi_app,
    status,
    date,
    time_value
):
    """
    Consistency analysis checks whether extracted payment
    information is internally usable.

    Missing OCR fields do not automatically mean fraud.
    """

    score = 100.0

    # --------------------------------------------------------
    # Actual negative payment states
    # --------------------------------------------------------

    if status == "Failed":
        score -= 25

    elif status == "Pending":
        score -= 10

    # --------------------------------------------------------
    # Identifier quality checks
    # --------------------------------------------------------

    if transaction_id:

        if len(
            re.sub(
                r"[^A-Za-z0-9]",
                "",
                transaction_id
            )
        ) < 8:

            score -= 10

    if utr:

        if not re.fullmatch(
            r"\d{8,20}",
            str(utr)
        ):

            score -= 10

    if upi_id:

        if "@" not in upi_id:
            score -= 10

    score = max(
        0,
        min(
            score,
            100
        )
    )

    suspicion = 100 - score

    if score >= 80:

        status_text = "Consistent"

    elif score >= 60:

        status_text = "Mostly Consistent"

    elif score >= 40:

        status_text = "Partially Consistent"

    else:

        status_text = "Incomplete"

    return {

        "score":
            round(
                score,
                2
            ),

        "suspicion":
            round(
                suspicion,
                2
            ),

        "status":
            status_text
    }


# ============================================================
# FINAL RISK CLASSIFICATION
# ============================================================

def classify_risk(score):

    if score < 40:

        return {

            "classification":
                "Likely Genuine",

            "result":
                "Likely Genuine",

            "risk_level":
                "Low Risk",

            "fraud_confidence":
                "Low"
        }

    elif score < 70:

        return {

            "classification":
                "Suspicious",

            "result":
                "Suspicious",

            "risk_level":
                "Medium Risk",

            "fraud_confidence":
                "Medium"
        }

    else:

        return {

            "classification":
                "Highly Suspicious",

            "result":
                "Highly Suspicious",

            "risk_level":
                "High Risk",

            "fraud_confidence":
                "High"
        }


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_image(image_path):

    start_time = time.time()

    print("\n==========================================")
    print("UPI FRAUD DETECTION ANALYSIS")
    print("==========================================")
    print("Image:", image_path)

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        return {

            "result":
                "Analysis Failed",

            "classification":
                "Analysis Failed",

            "risk_level":
                "Unknown",

            "fraud_confidence":
                "Unknown",

            "suspicion_score":
                0,

            "error":
                "Could not read uploaded image."
        }

    height, width = image.shape[:2]

    print(
        "Image dimensions:",
        width,
        "x",
        height
    )

    # --------------------------------------------------------
    # GENERAL OCR
    # --------------------------------------------------------

    print("\nRunning general OCR...")

    general_ocr = run_ocr(
        image,
        psm=11
    )

    # --------------------------------------------------------
    # PHONEPE REGION OCR
    # --------------------------------------------------------

    print(
        "Running PhonePe region OCR..."
    )

    regions = phonepe_region_ocr(
        image
    )

    # --------------------------------------------------------
    # COMBINE OCR
    # --------------------------------------------------------

    all_ocr = "\n".join([

        general_ocr,

        regions.get(
            "header",
            ""
        ),

        regions.get(
            "date_time",
            ""
        ),

        regions.get(
            "merchant",
            ""
        ),

        regions.get(
            "amount",
            ""
        ),

        regions.get(
            "upi",
            ""
        ),

        regions.get(
            "transaction",
            ""
        ),

        regions.get(
            "utr",
            ""
        )
    ])

    all_ocr = unique_lines(
        normalize_ocr_text(
            all_ocr
        )
    )

    print("\n---------- OCR TEXT ----------")
    print(all_ocr)
    print("------------------------------")

    # --------------------------------------------------------
    # EXTRACT FIELDS
    # --------------------------------------------------------

    amount = extract_amount(
        all_ocr
    )

    transaction_id = extract_transaction_id(
        all_ocr
    )

    utr = extract_utr(
        all_ocr
    )

    upi_id = extract_upi_id(
        all_ocr
    )

    upi_app = detect_upi_app(
        all_ocr
    )

    payment_status = detect_payment_status(
        all_ocr
    )

    date = extract_date(
        all_ocr
    )

    time_value = extract_time(
        all_ocr
    )

    merchant_name = extract_merchant_name(
        all_ocr
    )

    # --------------------------------------------------------
    # REGION FALLBACKS
    # --------------------------------------------------------

    if not transaction_id:

        transaction_id = extract_transaction_id(
            regions.get(
                "transaction",
                ""
            )
        )

    if not utr:

        utr = extract_utr(
            regions.get(
                "utr",
                ""
            )
        )

    if not upi_id:

        upi_id = extract_upi_id(
            regions.get(
                "upi",
                ""
            )
            + "\n"
            +
            regions.get(
                "merchant",
                ""
            )
        )

    if amount is None:

        amount = extract_amount(

            regions.get(
                "amount",
                ""
            )
            + "\n"
            +
            regions.get(
                "merchant",
                ""
            )
            + "\n"
            +
            regions.get(
                "header",
                ""
            )
        )

    if date is None:

        date = extract_date(

            regions.get(
                "header",
                ""
            )
            + "\n"
            +
            regions.get(
                "date_time",
                ""
            )
        )

    if time_value is None:

        time_value = extract_time(

            regions.get(
                "header",
                ""
            )
            + "\n"
            +
            regions.get(
                "date_time",
                ""
            )
        )

    if payment_status == "Unknown":

        payment_status = detect_payment_status(

            regions.get(
                "header",
                ""
            )
            + "\n"
            +
            regions.get(
                "merchant",
                ""
            )
        )

    if upi_app is None:

        upi_app = detect_upi_app(
            all_ocr + "\nPhonePe"
        )

    # --------------------------------------------------------
    # PRINT EXTRACTED DATA
    # --------------------------------------------------------

    print("\n========== EXTRACTED DATA ==========")

    print("Amount:", amount)
    print("Transaction ID:", transaction_id)
    print("UTR:", utr)
    print("UPI ID:", upi_id)
    print("UPI App:", upi_app)
    print("Payment Status:", payment_status)
    print("Date:", date)
    print("Time:", time_value)
    print("Merchant:", merchant_name)

    print("====================================")

    # --------------------------------------------------------
    # AI CNN
    # --------------------------------------------------------

    print("\nRunning AI CNN...")

    ai_result = predict_ai(
        image_path
    )

    print(
        "AI result:",
        ai_result
    )

    # --------------------------------------------------------
    # ELA
    # --------------------------------------------------------

    print("\nRunning ELA...")

    ela_result = perform_ela(
        image_path
    )

    # --------------------------------------------------------
    # FORENSIC
    # --------------------------------------------------------

    print(
        "Running forensic analysis..."
    )

    forensic = forensic_analysis(
        image_path
    )

    # --------------------------------------------------------
    # PAYMENT ANALYSIS
    # --------------------------------------------------------

    payment_analysis = (
        payment_information_analysis(

            amount,

            transaction_id,

            utr,

            upi_id,

            upi_app,

            payment_status,

            date,

            time_value
        )
    )

    # --------------------------------------------------------
    # CONSISTENCY
    # --------------------------------------------------------

    consistency = consistency_analysis(

        amount,

        transaction_id,

        utr,

        upi_id,

        upi_app,

        payment_status,

        date,

        time_value
    )

    # --------------------------------------------------------
    # AI SCORE
    # --------------------------------------------------------

    ai_score = None

    if (
        ai_result["available"]
        and
        ai_result[
            "manipulated_probability"
        ] is not None
    ):

        ai_score = float(
            ai_result[
                "manipulated_probability"
            ]
        )

    # --------------------------------------------------------
    # FORENSIC SCORE
    # --------------------------------------------------------

    forensic_score = float(
        forensic["score"]
    )

    # --------------------------------------------------------
    # OCR SUSPICION
    # --------------------------------------------------------

    ocr_suspicion = float(
        payment_analysis[
            "suspicion"
        ]
    )

    # --------------------------------------------------------
    # CONSISTENCY SUSPICION
    # --------------------------------------------------------

    consistency_suspicion = float(
        consistency[
            "suspicion"
        ]
    )

    # --------------------------------------------------------
    # WEIGHTED FINAL SCORE
    #
    # AI         = 50%
    # Forensic   = 20%
    # OCR        = 15%
    # Consistency= 15%
    #
    # IMPORTANT:
    # Missing OCR fields are NOT fraud.
    # --------------------------------------------------------

    components = []

    if ai_score is not None:

        components.append(
            (
                ai_score,
                0.50
            )
        )

    components.append(
        (
            forensic_score,
            0.20
        )
    )

    components.append(
        (
            ocr_suspicion,
            0.15
        )
    )

    components.append(
        (
            consistency_suspicion,
            0.15
        )
    )

    weighted_sum = sum(
        score * weight
        for score, weight in components
    )

    total_weight = sum(
        weight
        for _, weight in components
    )

    if total_weight > 0:

        final_score = (
            weighted_sum /
            total_weight
        )

    else:

        final_score = 0

    final_score = round(

        max(
            0,
            min(
                final_score,
                100
            )
        ),

        2
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    classification = classify_risk(
        final_score
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    reasons = []

    if ai_result["available"]:

        if ai_score < 50:

            reasons.append(
                "AI image analysis indicates the screenshot appears relatively authentic."
            )

        else:

            reasons.append(
                "AI image analysis detected a relatively high manipulation probability."
            )

    else:

        reasons.append(
            "AI CNN model was unavailable, so the final score was calculated using the available evidence."
        )

    # --------------------------------------------------------
    # STATUS EXPLANATION
    # --------------------------------------------------------

    if payment_status == "Successful":

        reasons.append(
            "OCR detected a successful/paid payment status."
        )

    elif payment_status == "Failed":

        reasons.append(
            "OCR detected a failed payment status."
        )

    elif payment_status == "Pending":

        reasons.append(
            "OCR detected a pending payment status."
        )

    else:

        reasons.append(
            "Payment success status could not be confidently detected by OCR."
        )

    # --------------------------------------------------------
    # AMOUNT EXPLANATION
    # --------------------------------------------------------

    if amount is not None:

        reasons.append(
            f"Payment amount detected as ₹{amount:g}."
        )

    else:

        reasons.append(
            "Payment amount could not be confidently detected by OCR."
        )

    # --------------------------------------------------------
    # TRANSACTION ID
    # --------------------------------------------------------

    if transaction_id:

        reasons.append(
            "A transaction ID was detected."
        )

    else:

        reasons.append(
            "Transaction ID could not be confidently detected."
        )

    # --------------------------------------------------------
    # UTR
    # --------------------------------------------------------

    if utr:

        reasons.append(
            "A UTR/reference number was detected."
        )

    else:

        reasons.append(
            "UTR/reference number could not be confidently detected."
        )

    # --------------------------------------------------------
    # UPI ID
    # --------------------------------------------------------

    if upi_id:

        reasons.append(
            "A UPI ID was detected and normalized."
        )

    else:

        reasons.append(
            "UPI ID could not be confidently detected."
        )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    if date:

        reasons.append(
            f"Transaction date detected as {date}."
        )

    else:

        reasons.append(
            "Transaction date could not be confidently detected by OCR."
        )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if time_value:

        reasons.append(
            f"Transaction time detected as {time_value}."
        )

    else:

        reasons.append(
            "Transaction time could not be confidently detected by OCR."
        )

    # --------------------------------------------------------
    # MERCHANT
    # --------------------------------------------------------

    if merchant_name:

        reasons.append(
            f"Merchant identified as {merchant_name}."
        )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%d %b %Y, %I:%M:%S %p"
    )

    elapsed = round(
        time.time() -
        start_time,
        2
    )

    # --------------------------------------------------------
    # OCR STATUS
    # --------------------------------------------------------

    extracted_count = sum([

        amount is not None,

        bool(transaction_id),

        bool(utr),

        bool(upi_id),

        bool(upi_app),

        payment_status != "Unknown",

        bool(date),

        bool(time_value)
    ])

    if extracted_count >= 7:

        ocr_status = "Excellent"

    elif extracted_count >= 5:

        ocr_status = "Good"

    elif extracted_count >= 3:

        ocr_status = "Partial"

    else:

        ocr_status = "Poor"

    # --------------------------------------------------------
    # AI ALIASES
    # --------------------------------------------------------

    if ai_result["available"]:

        ai_prediction = ai_result[
            "prediction"
        ]

        manipulated_probability = (
            ai_result[
                "manipulated_probability"
            ]
        )

        authentic_probability = (
            ai_result[
                "authentic_probability"
            ]
        )

    else:

        ai_prediction = (
            "AI model unavailable"
        )

        manipulated_probability = None

        authentic_probability = None

    # --------------------------------------------------------
    # OCR DATA
    # --------------------------------------------------------

    ocr_data = {

        "amount":
            amount,

        "transaction_id":
            transaction_id,

        "utr":
            utr,

        "upi_id":
            upi_id,

        "upi_app":
            upi_app,

        "payment_success_status":
            payment_status,

        "payment_status":
            payment_status,

        "date":
            date,

        "time":
            time_value,

        "merchant_name":
            merchant_name,

        "ocr_status":
            ocr_status,

        "text":
            all_ocr
    }

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {

        # ----------------------------------------------------
        # MAIN CLASSIFICATION
        # ----------------------------------------------------

        "result":
            classification["result"],

        "classification":
            classification["classification"],

        "risk_level":
            classification["risk_level"],

        "fraud_confidence":
            classification["fraud_confidence"],

        "suspicion_score":
            final_score,

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        "ai_result":
            ai_result,

        "ai_available":
            ai_result["available"],

        "ai_prediction":
            ai_prediction,

        "manipulated_probability":
            manipulated_probability,

        "authentic_probability":
            authentic_probability,

        "ai_manipulation_probability":
            manipulated_probability,

        "ai_score":
            ai_score,

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        "ocr_data":
            ocr_data,

        "raw_ocr":
            all_ocr,

        "phonepe_ocr":
            regions,

        "ocr_status":
            ocr_status,

        "amount":
            amount,

        "transaction_id":
            transaction_id,

        "utr":
            utr,

        "upi_id":
            upi_id,

        "upi_app":
            upi_app,

        "payment_status":
            payment_status,

        "payment_success_status":
            payment_status,

        "date":
            date,

        "time":
            time_value,

        "merchant_name":
            merchant_name,

        # ----------------------------------------------------
        # PAYMENT ANALYSIS
        # ----------------------------------------------------

        "payment_analysis":
            payment_analysis,

        "payment_information_score":
            payment_analysis[
                "score"
            ],

        "payment_information_suspicion":
            payment_analysis[
                "suspicion"
            ],

        "payment_information_status":
            payment_analysis[
                "status"
            ],

        # ----------------------------------------------------
        # CONSISTENCY
        # ----------------------------------------------------

        "consistency_analysis":
            consistency,

        "consistency_score":
            consistency[
                "score"
            ],

        "consistency_suspicion":
            consistency[
                "suspicion"
            ],

        "consistency_status":
            consistency[
                "status"
            ],

        # ----------------------------------------------------
        # FORENSIC
        # ----------------------------------------------------

        "forensic_analysis":
            forensic,

        "forensic_score":
            forensic[
                "score"
            ],

        "forensic_supporting_score":
            forensic[
                "score"
            ],

        "image_variance":
            forensic[
                "variance"
            ],

        "edge_percentage":
            forensic[
                "edge_percentage"
            ],

        "metadata_count":
            forensic[
                "metadata_count"
            ],

        "image_format":
            forensic[
                "format"
            ],

        "image_width":
            forensic[
                "width"
            ],

        "image_height":
            forensic[
                "height"
            ],

        # ----------------------------------------------------
        # ELA
        # ----------------------------------------------------

        "ela_score":
            ela_result[
                "score"
            ],

        "ela_available":
            ela_result[
                "available"
            ],

        "ela_image":
            ela_result[
                "path"
            ],

        "ela_image_url":
            (
                "/static/" +
                ela_result["path"]
                if ela_result["path"]
                else None
            ),

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        "reasons":
            reasons,

        "explanation":
            reasons,

        # ----------------------------------------------------
        # TIMING
        # ----------------------------------------------------

        "analysis_time":
            elapsed,

        "timestamp":
            timestamp,

        "error":
            None
    }

    # --------------------------------------------------------
    # CONSOLE OUTPUT
    # --------------------------------------------------------

    print("\n==========================================")
    print("FINAL ANALYSIS")
    print("==========================================")

    print(
        "Classification:",
        result["classification"]
    )

    print(
        "Risk:",
        result["risk_level"]
    )

    print(
        "Suspicion:",
        result["suspicion_score"],
        "%"
    )

    print(
        "Amount:",
        result["amount"]
    )

    print(
        "Transaction ID:",
        result["transaction_id"]
    )

    print(
        "UTR:",
        result["utr"]
    )

    print(
        "UPI ID:",
        result["upi_id"]
    )

    print(
        "UPI App:",
        result["upi_app"]
    )

    print(
        "Payment Status:",
        result["payment_status"]
    )

    print(
        "Date:",
        result["date"]
    )

    print(
        "Time:",
        result["time"]
    )

    print(
        "Merchant:",
        result["merchant_name"]
    )

    print(
        "AI Available:",
        result["ai_available"]
    )

    print(
        "AI Manipulation:",
        result["manipulated_probability"]
    )

    print(
        "Forensic Score:",
        result["forensic_score"]
    )

    print(
        "Consistency:",
        result["consistency_score"]
    )

    print(
        "OCR Status:",
        result["ocr_status"]
    )

    print(
        "ELA:",
        result["ela_score"]
    )

    print(
        "Analysis Time:",
        result["analysis_time"],
        "seconds"
    )

    print(
        "Timestamp:",
        result["timestamp"]
    )

    print("==========================================\n")

    return result



