from __future__ import annotations

import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass

from dataclasses import dataclass
from typing import Any, Dict
import logging
import os
import sys  # ← QO‘SHILDI

from joblib import load
import numpy as np

# ==== Joblib unpickling uchun dummy klass va patch ====
@dataclass
class TrainingConfig:
    """Joblib modeli ichida saqlangan konfiguratsiya uchun stub klass."""
    pass

# Model saqlanganda __main__.TrainingConfig sifatida yozilgan bo‘lgani uchun,
# hozirgi __main__ moduliga ham shu klassni qo‘shib qo‘yamiz.
sys.modules["__main__"].TrainingConfig = TrainingConfig
# ======================================================

@dataclass
class TrainingConfig:
    """Dummy class for loading pickled training_config.

    Eslatma: Bu klass faqat joblib load paytida kerak bo'ladi.
    Hamma real parametrlarga modelning o'zi javob beradi.
    """
    pass

# ----------------------
# Logger sozlamalari
# ----------------------
LOGGER = logging.getLogger("fraud_inference")
LOGGER.setLevel(logging.INFO)

if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    LOGGER.addHandler(handler)

# ----------------------
# Global o'zgaruvchilar
# ----------------------
MODEL: RandomForestClassifier | None = None
FEATURE_NAMES: List[str] = []

MODEL_PATH = "rf_fraud_model.joblib"

# Train fayldagi bilan bir xil mapping bo'lishi KERAK
TYPE_MAP = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}


def load_model(model_path: str = MODEL_PATH) -> None:
    """
    Joblib fayldan modelni va feature nomlarini yuklaydi.
    """
    global MODEL, FEATURE_NAMES

    LOGGER.info(f"Model yuklanmoqda: {model_path}")
    bundle = load(model_path)

    MODEL = bundle["model"]
    FEATURE_NAMES = bundle.get("feature_names", [])

    LOGGER.info("Model muvaffaqiyatli yuklandi. Feature soni: %d", len(FEATURE_NAMES))


def prepare_features_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inference paytida xom tranzaksiyani train paytidagi bilan
    bir xil formatga keltiradi.
    """

    # Keraksiz ustunlar (train dagi bilan bir xil bo'lsin)
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

    # 'type' ni kodlash
    if "type" in df.columns:
        df["type"] = df["type"].map(TYPE_MAP).fillna(-1).astype("int8")

    # NaNlarni 0 bilan to'ldiramiz
    df = df.fillna(0)

    # Raqamli ustunlarni float32 ga o'tkazamiz
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("float32")

    return df


def build_feature_frame(data: Dict[str, Any]) -> pd.DataFrame:
    """
    kelgan dict'dan bitta qatorli DataFrame quradi va
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
        f"Tranzaksiya baholandi. Fraud ehtimoli: {proba:.4f}, prediction: {pred}"
    )

    return {
        "prediction": pred,
        "fraud_probability": float(proba),
    }


# ----------------------
# Tezkor test
# ----------------------
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

