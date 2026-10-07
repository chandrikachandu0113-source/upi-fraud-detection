import os
import csv
import statistics
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "upi_transaction_history.csv"
)


def load_transaction_history():
    """
    Loads the synthetic/demo UPI transaction history.
    """

    if not os.path.exists(HISTORY_FILE):
        return []

    transactions = []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                try:
                    row["amount"] = float(row.get("amount", 0) or 0)
                except Exception:
                    row["amount"] = 0.0

                transactions.append(row)

    except Exception as e:
        print("Behaviour history error:", e)

    return transactions


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def normalize_text(value):
    if value is None:
        return ""

    return " ".join(str(value).strip().lower().split())


def extract_hour(time_value):
    """
    Extract hour from values such as:
    01:31
    01:31 PM
    13:31
    """

    if not time_value:
        return None

    value = str(time_value).strip().upper()

    formats = [
        "%I:%M %p",
        "%I:%M:%S %p",
        "%H:%M",
        "%H:%M:%S"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).hour
        except ValueError:
            continue

    return None


def calculate_amount_anomaly(amount, history):
    """
    Detects unusually large transaction amounts.

    Returns:
        score: 0-100
        message: explanation
    """

    if amount is None or amount <= 0:
        return 0.0, "Transaction amount unavailable for behaviour analysis."

    amounts = [
        float(row["amount"])
        for row in history
        if safe_float(row.get("amount")) is not None
        and float(row["amount"]) > 0
    ]

    if len(amounts) < 5:
        return 0.0, "Not enough historical transactions for amount analysis."

    median_amount = statistics.median(amounts)

    if median_amount <= 0:
        return 0.0, "Historical amount baseline unavailable."

    ratio = amount / median_amount

    # Normal range
    if ratio <= 3:
        return 0.0, "Transaction amount is within the normal historical range."

    # Moderately unusual
    if ratio <= 6:
        return 35.0, "Transaction amount is higher than the usual historical pattern."

    # Very unusual
    if ratio <= 10:
        return 65.0, "Transaction amount is significantly higher than the usual pattern."

    return 90.0, "Transaction amount is extremely unusual compared with historical transactions."


def calculate_frequency_anomaly(transaction_date, history):
    """
    Checks whether there are many transactions on the same day.
    """

    if not transaction_date:
        return 0.0, "Transaction date unavailable for frequency analysis."

    same_day = 0

    for row in history:
        if row.get("date") == transaction_date:
            same_day += 1

    if same_day <= 5:
        return 0.0, "Transaction frequency is within the normal range."

    if same_day <= 10:
        return 30.0, "Transaction frequency is somewhat higher than normal."

    if same_day <= 15:
        return 60.0, "High transaction frequency detected for this date."

    return 85.0, "Very high transaction frequency detected for this date."


def calculate_time_anomaly(time_value, history):
    """
    Detects unusual transaction time.

    Late-night transactions are given a moderate anomaly score,
    but time alone never marks a transaction as fraudulent.
    """

    hour = extract_hour(time_value)

    if hour is None:
        return 0.0, "Transaction time unavailable for behaviour analysis."

    historical_hours = []

    for row in history:
        h = extract_hour(row.get("time"))

        if h is not None:
            historical_hours.append(h)

    if not historical_hours:
        return 0.0, "No historical time pattern available."

    # Late night / very early morning
    if 0 <= hour < 5:
        return 35.0, "Transaction occurred during an unusual late-night/early-morning period."

    return 0.0, "Transaction time is within a normal usage period."


def calculate_merchant_pattern(merchant, history):
    """
    Checks whether the merchant is commonly seen.

    A new merchant is NOT automatically treated as fraud.
    """

    merchant = normalize_text(merchant)

    if not merchant:
        return 0.0, "Merchant unavailable for behaviour analysis."

    merchants = [
        normalize_text(row.get("merchant"))
        for row in history
        if row.get("merchant")
    ]

    if not merchants:
        return 0.0, "No historical merchant pattern available."

    if merchant in merchants:
        return 0.0, "Merchant is present in the historical transaction pattern."

    return 5.0, "New merchant detected; this is only a weak behavioural signal."


def calculate_upi_pattern(upi_id, history):
    """
    Checks whether the UPI ID has appeared before.

    A new UPI ID is only a weak signal.
    """

    upi_id = normalize_text(upi_id)

    if not upi_id:
        return 0.0, "UPI ID unavailable for behaviour analysis."

    upi_values = [
        normalize_text(row.get("upi_id"))
        for row in history
        if row.get("upi_id")
    ]

    if upi_id in upi_values:
        return 0.0, "UPI ID is present in the historical transaction pattern."

    return 5.0, "New UPI ID detected; this is only a weak behavioural signal."


def classify_behavior(score):
    """
    Converts behaviour score into a readable classification.
    """

    if score < 25:
        return "Normal Behaviour", "Low"

    if score < 50:
        return "Slightly Unusual Behaviour", "Medium"

    if score < 70:
        return "Unusual Behaviour", "Medium"

    return "Highly Unusual Behaviour", "High"


def analyze_behavior(
    amount=None,
    merchant=None,
    upi_id=None,
    date=None,
    time=None,
    payment_status=None
):
    """
    Main behaviour-pattern analysis function.

    The function is deliberately explainable and suitable
    for a final-year project demonstration.
    """

    history = load_transaction_history()

    if not history:
        return {
            "available": False,
            "score": 0.0,
            "risk_level": "Unknown",
            "classification": "Behaviour Analysis Unavailable",
            "alert": False,
            "alert_level": "None",
            "reasons": [
                "No transaction history dataset was found."
            ],
            "message": "Behaviour analysis could not be performed."
        }

    amount_score, amount_message = calculate_amount_anomaly(
        safe_float(amount),
        history
    )

    frequency_score, frequency_message = calculate_frequency_anomaly(
        date,
        history
    )

    time_score, time_message = calculate_time_anomaly(
        time,
        history
    )

    merchant_score, merchant_message = calculate_merchant_pattern(
        merchant,
        history
    )

    upi_score, upi_message = calculate_upi_pattern(
        upi_id,
        history
    )

    # Payment failure is an additional behavioural warning.
    status_score = 0.0
    status_message = "Payment status does not indicate behavioural risk."

    status = normalize_text(payment_status)

    if status == "failed":
        status_score = 40.0
        status_message = "Failed payment status detected."

    elif status == "pending":
        status_score = 20.0
        status_message = "Pending payment status detected."

    elif status in ("successful", "completed"):
        status_score = 0.0
        status_message = "Successful payment status detected."

    # Weighted behavioural score.
    score = (
        amount_score * 0.40 +
        frequency_score * 0.20 +
        time_score * 0.15 +
        merchant_score * 0.10 +
        upi_score * 0.05 +
        status_score * 0.10
    )

    score = round(min(max(score, 0.0), 100.0), 2)

    classification, risk_level = classify_behavior(score)

    # Real-time alert decision.
    if score >= 70:
        alert = True
        alert_level = "HIGH"
        alert_message = (
            "Real-time alert: highly unusual transaction behaviour detected. "
            "Verify the transaction using the official bank/UPI transaction history."
        )

    elif score >= 50:
        alert = True
        alert_level = "MEDIUM"
        alert_message = (
            "Real-time warning: unusual transaction behaviour detected. "
            "Additional verification is recommended."
        )

    else:
        alert = False
        alert_level = "NONE"
        alert_message = (
            "No significant behavioural anomaly detected."
        )

    reasons = [
        amount_message,
        frequency_message,
        time_message,
        merchant_message,
        upi_message,
        status_message
    ]

    return {
        "available": True,
        "score": score,
        "risk_level": risk_level,
        "classification": classification,
        "alert": alert,
        "alert_level": alert_level,
        "alert_message": alert_message,
        "reasons": reasons,
        "history_records": len(history),

        "amount_score": round(amount_score, 2),
        "frequency_score": round(frequency_score, 2),
        "time_score": round(time_score, 2),
        "merchant_score": round(merchant_score, 2),
        "upi_score": round(upi_score, 2),
        "status_score": round(status_score, 2),

        "amount": amount,
        "merchant": merchant,
        "upi_id": upi_id,
        "date": date,
        "time": time,
        "payment_status": payment_status,

        "message": "Behaviour pattern analysis completed successfully."
    }

