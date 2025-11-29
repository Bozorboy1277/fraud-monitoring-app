# ============================================
# 1-BLOK: Loyihaning boshlang'ich sozlamalari
# ============================================

from __future__ import annotations

import os
import random
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)


# Agar grafiklar kerak bo'lsa (keyinroq ishlatamiz)



# ------------------------
# Tasodifiylikni boshqarish
# ------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Barcha asosiy kutubxonalar uchun random urug'ni o'rnatadi,
    natijalar qayta tiklanuvchi (reproducible) bo'lishi uchun.
    """
    random.seed(seed)
    np.random.seed(seed)
    # sklearn o'zida random_state parametridan foydalanamiz


set_global_seed(42)


# -------------------
# Logging konfiguratsiya
# -------------------

def init_logger(
    name: str = "fraud_detection",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Loyihada foydalanish uchun bitta umumiy logger yaratadi.
    Konsolga chiroyli formatda log yozib boradi.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Agar handler allaqachon qo'shilgan bo'lsa, qayta qo'shmaymiz
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


LOGGER = init_logger()


# -------------------
# Umumiy konfiguratsiya
# -------------------

@dataclass
class DatasetConfig:
    """
    Har bir Kaggle dataset uchun yo'l va ustun nomlarini
    boshqarish uchun konfiguratsiya.
    """
    name: str
    path: str
    label_column: str
    # Keyinroq mappinglar qo'shamiz (masalan, amount, time, va hokazo)
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    """
    RandomForest va train-test bo'linishi uchun asosiy sozlamalar.
    """
    test_size: float = 0.2
    val_size: float = 0.1   # istasak train ichidan validation ajratamiz
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None
    n_jobs: int = -1
    class_weight: str = "balanced"   # fraud klaslari notekis bo'lgani uchun
    # Thresholdni keyinchalik risk_score bo'yicha sozlaymiz
    decision_threshold: float = 0.5


# Bu konfiguratsiyani keyin train bosqichida ishlatamiz
TRAINING_CONFIG = TrainingConfig()


# ----------------------
# Fayl bor-yo'qligini tekshirish
# ----------------------

def check_file_exists(path: str) -> None:
    """
    Dataset faylini yuklashdan oldin uning mavjudligini tekshiradi.
    """
    if not os.path.exists(path):
        LOGGER.error(f"Fayl topilmadi: {path}")
        raise FileNotFoundError(f"Dataset fayli topilmadi: {path}")
    LOGGER.info(f"Dataset fayli topildi: {path}")

  # ============================================
# 2-BLOK: Dataset konfiguratsiyasi va yuklash
# ============================================

from typing import List


# --------------------------
# 2.1. Kaggle datasetlar config'i
# --------------------------

DATASETS: List[DatasetConfig] = [
    DatasetConfig(
        name="FraudDataset1",
        path="Fraud.csv",          # kerak bo'lsa to'liq yo'lni yozing
        label_column="isFraud",
        extra_params={
            "rename": {
                # amount allaqachon to'g'ri nomda
                "step": "timestamp",   # vaqt o'rnida ishlatamiz
            }
        },
    ),
    DatasetConfig(
        name="CreditCardDataset",
        path="creditcard.csv",     # kerak bo'lsa to'liq yo'lni yozing
        label_column="Class",
        extra_params={
            "rename": {
                "Amount": "amount",
                "Time": "timestamp",
            }
        },
    ),
    # ❗ bank_transactions_data_2.csv da hozircha fraud label yo'q,
    # shu sababli uni supervised trainingga qo'shmayapmiz.
    # Keyinroq alohida ishlatish mumkin bo'ladi.
     DatasetConfig(
         name="BankTransactionsRaw",
         path="bank_transactions_data_2.csv",
         label_column="...",  # agar label ustun qo'shsak
         extra_params={
             "rename": {
                 "TransactionAmount": "amount",
                 "TransactionDate": "timestamp",
             }
         },
     ),
]


# --------------------------
# 2.2. Bitta datasetni yuklash
# --------------------------

def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    """
    Kaggle datasetni yuklaydi, label ustunini 'is_fraud' deb nomlaydi,
    kerak bo'lsa ustunlarni qayta nomlaydi va minimal tekshiruvdan o'tkazadi.
    """

    # Fayl mavjudligini tekshiramiz
    check_file_exists(config.path)

    LOGGER.info(f"[{config.name}] dataset yuklanmoqda: {config.path}")
    df = pd.read_csv(config.path)

    LOGGER.info(f"[{config.name}] o'qildi. Shape: {df.shape}")

    # Label ustuni mavjudligini tekshiramiz
    if config.label_column not in df.columns:
        raise ValueError(
            f"[{config.name}] datasetda '{config.label_column}' ustuni topilmadi!"
        )

    # Label ustunini yagona nomga o'tkazamiz
    df = df.rename(columns={config.label_column: "is_fraud"})

    # is_fraud ni 0/1 ko'rinishga majburlab keltiramiz
    # (ba'zi datasetlarda True/False yoki float bo'lishi mumkin)
    df["is_fraud"] = df["is_fraud"].astype(int)

    # Agar qo'shimcha nomlash (rename) bo'lsa, qo'llaymiz
    if config.extra_params and "rename" in config.extra_params:
        df = df.rename(columns=config.extra_params["rename"])

    # Minimal zarur ustunlar ro'yxati
    required_cols = ["amount", "is_fraud"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"[{config.name}] datasetda '{col}' ustuni topilmadi. "
                f"extra_params['rename'] konfiguratsiyasini tekshiring."
            )

    # Qo'shimcha informatsiya uchun manba nomini ham qo'shib qo'yamiz
    df["source"] = config.name

    LOGGER.info(
        f"[{config.name}] tayyorlandi. Fraud ulushi: "
        f"{df['is_fraud'].mean() * 100:.4f}%"
    )

    return df


# --------------------------
# 2.3. Bir nechta datasetni birlashtirish
# --------------------------

def merge_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Har xil Kaggle datasetlardan olingan dataframe'larni bitta umumiy DFga birlashtiradi.
    """
    if not datasets:
        raise ValueError("Birlashtirish uchun hech qanday dataset kelmadi (bo'sh ro'yxat).")

    LOGGER.info(f"{len(datasets)} ta dataset birlashtirilmoqda...")

    merged = pd.concat(datasets, axis=0, ignore_index=True)

    LOGGER.info(f"Birlashgan dataset shape: {merged.shape}")
    LOGGER.info(
        "Umumiy fraud ulushi (label=1): "
        f"{merged['is_fraud'].mean() * 100:.4f}%"
    )

    return merged

def sample_large_dataset(df: pd.DataFrame, max_rows: int = 500_000) -> pd.DataFrame:
    """
    Katta datasetni max_rows gacha kamaytiradi.
    Hackaton uchun kompyuter resurslarini tejashga yordam beradi.
    Hozircha oddiy random sampling ishlatyapmiz.
    """
    n_rows = len(df)
    if n_rows <= max_rows:
        LOGGER.info(
            f"Dataset hajmi {n_rows} qator. Sampling talab qilinmaydi."
        )
        return df

    LOGGER.info(
        f"Dataset juda katta: {n_rows} qator. "
        f"Random sampling orqali {max_rows} qatorkacha qisqartiriladi."
    )

    df_sampled = df.sample(
        n=max_rows,
        random_state=TRAINING_CONFIG.random_state
    )

    LOGGER.info(f"Samplingdan keyingi shape: {df_sampled.shape}")
    return df_sampled

# --------------------------
# 2.4. Hammasini yuklab, bitta DFga yig'ish
# --------------------------

def load_all_labeled_data() -> pd.DataFrame:
    """
    DATASETS ro'yxatidagi barcha labelga ega datasetlarni yuklaydi,
    xatolik bo'lsa logger orqali xabar beradi,
    va muvaffaqiyatli yuklangan datasetlarni bitta DFga birlashtiradi.
    Shu yerning o'zida katta datasetni sampling qilamiz.
    """
    loaded: List[pd.DataFrame] = []

    for cfg in DATASETS:
        try:
            df = load_dataset(cfg)
            loaded.append(df)
        except Exception as exc:
            LOGGER.error(f"[{cfg.name}] yuklashda xato: {exc}")

    if not loaded:
        raise RuntimeError("Hech bir dataset muvaffaqiyatli yuklanmadi.")

    merged_df = merge_datasets(loaded)

    # !!! MUHIM: Katta datasetni shu yerning o'zida kichraytiramiz
    merged_df = sample_large_dataset(merged_df, max_rows=500_000)

    return merged_df



# 2.5. Tezkor test (xohlasangiz ishlatib ko'ring)
if __name__ == "__main__":
    try:
        all_data = load_all_labeled_data()
        LOGGER.info(all_data.head().to_string())
    except Exception as e:
        LOGGER.error(f"Yuklash jarayonida xato: {e}")


# ============================================
# 3-BLOK: Feature engineering va train/test ajratish
# ============================================

from sklearn.model_selection import train_test_split


TYPE_MAP = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model uchun kerakli feature'larni tayyorlash:
    - Keraksiz ustunlarni olib tashlash
    - 'type' ustunini sonli formatga o'tkazish (oldindan berilgan mapping)
    - NaNlarni to'ldirish
    - Raqamli ustunlarni float32 ga o'tkazish (xotiraga yengilroq)
    """

    LOGGER.info("Feature engineering boshlanmoqda...")

    # 1) Keraksiz ustunlar
    drop_cols = [
        "nameOrig",
        "nameDest",
        "source",
        "isFlaggedFraud",
    ]

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 2) 'type' ustunini oldindan belgilangan kodlarga o'tkazamiz
    if "type" in df.columns:
        df["type"] = df["type"].map(TYPE_MAP).fillna(-1).astype("int8")

    # 3) NaNlarni to'ldirish
    df = df.fillna(0)

    # 4) Raqamli ustunlarni float32 ga o'tkazamiz (RAMni tejash uchun)
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("float32")

    LOGGER.info(f"Feature engineering tugadi. Yakuniy shape: {df.shape}")
    return df

    
    # 3.3. NaNlarni to'ldirish
    df = df.fillna(0)

    LOGGER.info(f"Feature engineering tugadi. Yakuniy shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    """
    Train/test split qiladi.
    """
    LOGGER.info("Train/test bo'linmoqda...")

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAINING_CONFIG.test_size,
        random_state=TRAINING_CONFIG.random_state,
        stratify=y,
    )

    LOGGER.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ============================================
# 4-BLOK: RandomForest modelni o'qitish
# ============================================

from joblib import dump  # modelni saqlash uchun


def train_random_forest(
    df_raw: pd.DataFrame,
    model_path: str = "rf_fraud_model.joblib",
) -> RandomForestClassifier:
    """
    To'liq training pipeline:
    - xom datasetdan feature engineering
    - train/test split
    - RandomForestClassifierni o'qitish
    - metriku va natijalarni logga chiqarish
    - modelni faylga saqlash (MVP uchun juda muhim)

    :param df_raw: load_all_labeled_data() dan olingan umumiy dataframe
    :param model_path: saqlanadigan model fayli nomi
    :return: o'qitilgan RandomForestClassifier obyekti
    """

    LOGGER.info("=== TRAINING PIPELINE BOSHLANDI (RandomForest) ===")

    # 1) Feature engineering
    df = prepare_features(df_raw)

    # 2) Train/test bo'linishi
    X_train, X_test, y_train, y_test = split_data(df)

    # 3) Katta datasetlar uchun sampling (kompyuter zo'riqmasin)
    #    Agar train juda katta bo'lsa, bir qismini olishimiz mumkin
    max_train_samples = 500_000  # kerak bo'lsa kamaytirish mumkin

    if len(X_train) > max_train_samples:
        LOGGER.info(
            f"Train hajmi {len(X_train)} ta. "
            f"{max_train_samples} tasini stratified sampling bilan olamiz."
        )

        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=max_train_samples,
            random_state=TRAINING_CONFIG.random_state,
            stratify=y_train,
        )

        LOGGER.info(f"Samplingdan keyingi train shape: {X_train.shape}")

    # 4) Modelni yaratish
    LOGGER.info("RandomForestClassifier yaratilyapti...")

    model = RandomForestClassifier(
        n_estimators=TRAINING_CONFIG.n_estimators,
        max_depth=TRAINING_CONFIG.max_depth,
        n_jobs=TRAINING_CONFIG.n_jobs,
        class_weight=TRAINING_CONFIG.class_weight,
        random_state=TRAINING_CONFIG.random_state,
    )

    # 5) Modelni o'qitish
    LOGGER.info("Model training boshlanmoqda...")
    model.fit(X_train, y_train)
    LOGGER.info("Model training tugadi.")

    # 6) Test setda baholash
    LOGGER.info("Model test setda baholanmoqda...")

    y_pred = model.predict(X_test)
    # ehtimollar (fraud bo'lish ehtimoli)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        auc = None

    report_text = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    LOGGER.info("=== CLASSIFICATION REPORT ===")
    LOGGER.info("\n" + report_text)

    LOGGER.info("=== CONFUSION MATRIX ===")
    LOGGER.info("\n%s", cm)

    if auc is not None:
        LOGGER.info("ROC–AUC: %.4f", auc)

    # 7) Modelni faylga saqlash
    if model_path:
        dump(
            {
                "model": model,
                "feature_names": X_train.columns.tolist(),
                "training_config": TRAINING_CONFIG,
            },
            model_path,
        )
        LOGGER.info(f"Model saqlandi: {model_path}")

    LOGGER.info("=== TRAINING PIPELINE YAKUNLANDI ===")
    return model


# --------------------------------------------
# 4.1. Barcha pipeline'ni ishga tushirish (main)
# --------------------------------------------

if __name__ == "__main__":
    try:
        # 1) Barcha labelga ega datasetlarni yuklaymiz (va ichida sampling ham bo'ladi)
        all_data = load_all_labeled_data()

        # 2) RandomForest modelni o'qitamiz
        trained_model = train_random_forest(
            all_data,
            model_path="rf_fraud_model.joblib",
        )

        LOGGER.info("Pipeline muvaffaqiyatli tugadi.")
    except Exception as e:
        LOGGER.error(f"Yuklash / training jarayonida xato: {e}")