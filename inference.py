from __future__ import annotations

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
import sys

import numpy as np
import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# 1. Joblib unpickling uchun TrainingConfig stub + patch
# =========================================================

@dataclass
class TrainingConfig:
    """
    Joblib orqali saqlangan model ichida 'TrainingConfig' obyekti bor.
    Bizga faqat unpickling paytida nomi kerak bo'ladi, shuning uchun
    bo'sh stub klass kifoya.
    """
    pass


# Model saqlanganda __main__.TrainingConfig sifatida yozilgan bo'lgan.
# Lokalda __main__ = 'Untitled-1.py' bo'lgan, Renderda esa __main__ = 'gunicorn'.
# Shuning uchun hozirgi __main__ moduliga ham TrainingConfig ni biriktirib qo'yamiz.
sys.modules["__main__"].TrainingConfig = TrainingConfig


# =========================================================
# 2. Logger sozlamalari
# =========================================================

LOGGER = logging.getLogger("fraud_inference")
LOGGER.setLevel(logging.INFO)

if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    LOGGER.addHandler(handler)


# =========================================================
# 3. Global o'zgaruvchilar va mapping
# =========================================================

MODEL: RandomForestClassifier | None = None
FEATURE_NAMES: List[str] = []

MODEL_PATH = "rf_fraud_model.joblib"

# Train kodida ishlatilgan mapping bilan BIR xil bo'lishi shart
TYPE_MAP: Dict[str, int] = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}


# =========================================================
# 4. Modelni yuklash
# =========================================================

def load_model(model_path: str = MODEL_PATH) -> None:
    """
    Joblib fayldan model va feature nomlarini yuklaydi.
    """
    global MODEL, FEATURE_NAMES

    LOGGER.info("Model yuklanmoqda: %s", model_path)
    bundle = load(model_path)

    MODEL = bundle["model"]
    FEATURE_NAMES = bundle.get("feature_names", [])

    LOGGER.info(
        "Model muvaffaqiyatli yuklandi. Feature soni: %d",
        len(FEATURE_NAMES),
    )


# =========================================================
# 5. Feature tayyorlash (inference uchun)
# =========================================================

def prepare_features_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inference paytida xom tranzaksiyani train paytidagi bilan
    bir xil formatga keltiradi.
    """
    drop_cols = [
        "nameOrig",
        "nameDest",
        "source",
        "isFlaggedFraud",
        "is_fraud",  # ehtiyotan, bo'lsa olib tashlaymiz
    ]

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 'type' ni raqamga map qilish
    if "type" in df.columns:
        df["type"] = df["type"].map(TYPE_MAP).fillna(-1).astype("int8")

    # NaNlarni 0 bilan to'ldiramiz
    df = df.fillna(0)

    # Raqamli ustunlarni float32 ga o'tkazamiz (tezroq va kamroq RAM)
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("float32")

    return df


def build_feature_frame(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Kelgan dict'dan bitta qatorli DataFrame quradi va
    train paytida ishlatilgan FEATURE_NAMES tartibiga moslashtiradi.
    """
    if MODEL is None or not FEATURE_NAMES:
        load_model()

    # 1-qatorli DF yaratamiz
    raw_df = pd.DataFrame([data])

    # Train dagi kabi feature engineering
    df = prepare_features_inference(raw_df)

    # Yetishmayotgan ustunlarni 0 bilan qo'shamiz
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0

    # Ortiqcha ustunlar bo'lsa, ularni tashlab yuboramiz
    df = df[FEATURE_NAMES]

    return df


# =========================================================
# 6. Bashorat funksiyasi
# =========================================================

def predict_transaction(data: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Bitta tranzaksiya bo'yicha fraud / normal bashorat qiladi.

    :param data: masalan:
        {
            "timestamp": 1,
            "type": "PAYMENT",
            "amount": 90000000,
            "oldbalanceOrg": 100.0,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 90000000,
        }
    :param threshold: fraud ehtimoli qaysi chegaradan boshlab 1 deb olinadi
    :return: {"prediction": 0/1, "fraud_probability": float}
    """
    if MODEL is None:
        load_model()

    X = build_feature_frame(data)

    proba = MODEL.predict_proba(X)[0, 1]
    pred = int(proba >= threshold)

    LOGGER.info(
        "Tranzaksiya baholandi. Fraud ehtimoli: %.4f, prediction: %d",
        proba,
        pred,
    )

    return {
        "prediction": pred,
        "fraud_probability": float(proba),
    }


# =========================================================
# 7. Tezkor lokal test
# =========================================================

if __name__ == "__main__":
    sample_tx = {
        "timestamp": 1.0,
        "type": "PAYMENT",
        "amount": 90000000.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 90000000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "M987654321",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 90000000.0,
        # qolgan ustunlar yo'q bo'lsa ham bo'ladi — 0 bilan to'ldiriladi
    }

    result = predict_transaction(sample_tx, threshold=0.5)
    print("Natija:", result)
