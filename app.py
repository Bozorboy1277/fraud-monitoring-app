from __future__ import annotations

# ================================================
# 1. IMPORTLAR VA BAZAVIY SOZLAMALAR
# ================================================
import logging
from typing import Any, Dict, Tuple, List

from dataclasses import dataclass
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import BadRequest, HTTPException

from inference import predict_transaction, TYPE_MAP  # ML inference funksiyasi


# Joblib unpickling uchun dummy klass
@dataclass
class TrainingConfig:
    """Dummy class for unpickling training_config from joblib."""
    pass


LOGGER = logging.getLogger("fraud_api")
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    LOGGER.addHandler(handler)


# Tranzaksiya historiyasi (oxirgi 10 ta)
HISTORY: List[Dict[str, Any]] = []


# ================================================
# 2. YORDAMCHI FUNKSIYALAR: JAVOB FORMATLASH
# ================================================
def build_success_response(
    data: Dict[str, Any],
    status_code: int = 200,
) -> Tuple[Any, int]:
    payload = {
        "success": True,
        "data": data,
    }
    return jsonify(payload), status_code


def build_error_response(
    message: str,
    status_code: int = 400,
    details: Dict[str, Any] | None = None,
) -> Tuple[Any, int]:
    payload: Dict[str, Any] = {
        "success": False,
        "error": {
            "message": message,
            "code": status_code,
        },
    }
    if details:
        payload["error"]["details"] = details
    return jsonify(payload), status_code


# ================================================
# 3. REQUEST VALIDATSIYASI (API /predict uchun)
# ================================================
REQUIRED_FIELDS = [
    "type",
    "amount",
]

NUMERIC_FIELDS = [
    "timestamp",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def validate_and_normalize_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Kiruvchi JSON body'ni tekshiradi va modelga tayyorlaydi.
    (API /predict uchun ishlatiladi)
    """
    if not isinstance(raw, dict):
        raise BadRequest("JSON body noto'g'ri formatda (dict bo'lishi kerak).")

    # Kerakli maydonlar
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise BadRequest(f"Kerakli maydonlar yetishmayapti: {', '.join(missing)}")

    # type qiymati
    tx_type = raw.get("type")
    if tx_type not in TYPE_MAP:
        allowed = ", ".join(TYPE_MAP.keys())
        raise BadRequest(
            f"Yaroqsiz 'type' qiymati: {tx_type}. Ruxsat etilganlar: {allowed}"
        )

    # Raqamli maydonlar
    normalized: Dict[str, Any] = dict(raw)
    for field in NUMERIC_FIELDS:
        if field in normalized and normalized[field] is not None:
            try:
                normalized[field] = float(normalized[field])
            except (TypeError, ValueError):
                raise BadRequest(f"'{field}' maydoni raqam bo'lishi kerak.")

    if "timestamp" not in normalized:
        normalized["timestamp"] = 0.0

    return normalized


# ================================================
# 4. FLASK APP YARATISH (APP FACTORY)
# ================================================
def create_app() -> Flask:
    app = Flask(__name__)

    # ---------- Error handlerlar ----------

    @app.errorhandler(BadRequest)
    def handle_bad_request(e: BadRequest):
        LOGGER.warning(f"BadRequest: {e}")
        return build_error_response(str(e), status_code=400)

    @app.errorhandler(HTTPException)
    def handle_http_exception(e: HTTPException):
        LOGGER.error(f"HTTPException: {e}")
        return build_error_response(e.description, status_code=e.code or 500)

    @app.errorhandler(Exception)
    def handle_unexpected_error(e: Exception):
        LOGGER.exception("Kutilmagan server xatosi:")
        # Dev rejimida xatoni to'liq ko'rsatamiz (hackaton uchun qulay)
        return build_error_response(f"Server xatosi: {e}", status_code=500)

    # ---------- UI ROUTELAR ----------

    @app.get("/")
    def index():
        """
        Asosiy web sahifa (UI).
        tx_types -> select uchun tranzaksiya turlari.
        history -> so'nggi 10 tranzaksiya.
        """
        return render_template(
            "index.html",
            tx_types=list(TYPE_MAP.keys()),
            result=None,
            error=None,
            history=HISTORY,
        )

    @app.get("/api-info")
    def api_info_page():
        """
        API haqida HTML formatdagi qisqacha dokumentatsiya sahifasi.
        """
        return render_template("api_info.html")

    @app.get("/health-page")
    def health_page():
        """
        Health statusni vizual ko'rsatadigan sahifa.
        """
        return render_template(
            "health.html",
            status="online",
            last_check=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    @app.post("/ui/predict")
    def ui_predict():
        """
        Web forma orqali yuborilgan tranzaksiyani baholaydi
        va natijani yana index sahifasida ko'rsatadi.
        """
        form = request.form
        error_message = None
        view_model: Dict[str, Any] | None = None

        try:
            tx_type = form.get("type") or "PAYMENT"
            amount = float(form.get("amount") or 0)
            oldbalance_org = float(form.get("oldbalanceOrg") or 0)
            newbalance_orig = float(form.get("newbalanceOrig") or 0)
            oldbalance_dest = float(form.get("oldbalanceDest") or 0)
            newbalance_dest = float(form.get("newbalanceDest") or 0)
            timestamp = float(form.get("timestamp") or 0)

            payload = {
                "timestamp": timestamp,
                "type": tx_type,
                "amount": amount,
                "oldbalanceOrg": oldbalance_org,
                "newbalanceOrig": newbalance_orig,
                "oldbalanceDest": oldbalance_dest,
                "newbalanceDest": newbalance_dest,
            }

            # UI uchun biroz pastroq threshold – sezgirroq qilsin:
            risk_threshold = 0.15  # 15% dan yuqori bo'lsa fraud deb olamiz

            # ML modelini chaqiramiz
            result = predict_transaction(payload, threshold=risk_threshold)

            view_model = {
                "input": payload,
                "prediction": int(result["prediction"]),
                "fraud_probability": float(result["fraud_probability"]),
                "threshold": risk_threshold,
            }

            # ---- HISTORY GA YOZISH ----
            global HISTORY
            HISTORY.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": tx_type,
                    "amount": amount,
                    "fraud_probability": view_model["fraud_probability"],
                    "prediction": view_model["prediction"],
                }
            )
            # faqat oxirgi 10 tasini saqlaymiz
            HISTORY = HISTORY[-10:]

        except ValueError:
            error_message = "Raqamli maydonlarga faqat son kiritish mumkin."
        except Exception as e:
            error_message = f"Server xatosi: {e}"

        return render_template(
            "index.html",
            tx_types=list(TYPE_MAP.keys()),
            result=view_model,
            error=error_message,
            history=HISTORY,
        )

    # ---------- API ROUTELAR ----------

    @app.get("/health")
    def health_check():
        """
        Health-check endpoint (JSON).
        """
        return build_success_response({"status": "ok"})

    @app.get("/meta")
    def meta():
        """
        API haqida qisqacha meta ma'lumot (JSON).
        """
        return build_success_response(
            {
                "service": "Fraud Detection API",
                "version": "1.0.0",
                "allowed_transaction_types": list(TYPE_MAP.keys()),
                "description": "Bank tranzaksiyalarida firibgarlikni aniqlash uchun ML modeli asosidagi API.",
            }
        )

    @app.post("/predict")
    def predict():
        """
        JSON API: Fraud / normal bashorati uchun endpoint.
        """
        try:
            payload = request.get_json(silent=False)
        except BadRequest:
            raise BadRequest("JSON body noto'g'ri yoki bo'sh.")

        if payload is None:
            raise BadRequest("JSON body topilmadi.")

        LOGGER.info(f"/predict ga so'rov keldi: {payload}")

        raw_threshold = payload.pop("threshold", 0.5)
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            raise BadRequest("'threshold' raqam bo'lishi kerak.")

        normalized = validate_and_normalize_payload(payload)

        result = predict_transaction(normalized, threshold=threshold)

        response_data = {
            "input": normalized,
            "prediction": int(result["prediction"]),
            "fraud_probability": float(result["fraud_probability"]),
            "threshold": threshold,
        }

        return build_success_response(response_data)

    return app


# ======== MUHIM QO‘SHIMCHA QATOR ========
# Gunicorn (app:app) uchun global Flask instance:
app = create_app()
# ========================================


# ================================================
# 5. ENTRY POINT
# ================================================
if __name__ == "__main__":
    # Lokal rejimda ham xuddi shu app'ni ishlatamiz
    app.run(host="0.0.0.0", port=8000, debug=True)
