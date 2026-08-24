"""Pemuatan model, penskoran, dan perhitungan metrik evaluasi."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from . import config as C
from .data import siapkan_X


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    """Pipeline final: RobustScaler + Ordinal/OneHot encoder + HistGradientBoosting."""
    with open(C.MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def info_model() -> dict:
    """Ringkasan struktur pipeline untuk halaman dokumentasi model."""
    model = muat_model()
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    kategori_nominal = dict(zip(
        prep.transformers_[2][2],
        [list(c) for c in prep.named_transformers_["nom"].categories_],
    ))
    return {
        "algoritma": type(clf).__name__,
        "hyperparameter": {
            "learning_rate": clf.learning_rate,
            "max_iter": clf.max_iter,
            "max_leaf_nodes": clf.max_leaf_nodes,
            "min_samples_leaf": clf.min_samples_leaf,
            "l2_regularization": clf.l2_regularization,
        },
        "n_fitur_masuk": len(C.FITUR_MODEL),
        "n_kolom_setelah_encoding": len(prep.get_feature_names_out()),
        "kategori_nominal": kategori_nominal,
    }


def skor(df_fitur: pd.DataFrame) -> np.ndarray:
    """Probabilitas nasabah membuka deposito. `df_fitur` sudah lewat buat_fitur()."""
    model = muat_model()
    return model.predict_proba(siapkan_X(df_fitur))[:, 1]


# --------------------------------------------------------------------------- #
# Evaluasi pada data uji (20% yang tidak pernah dipakai melatih model)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Menghitung performa model pada data uji...")
def evaluasi_data_uji(_df: pd.DataFrame) -> dict:
    """Ulangi pembagian 80/20 notebook (stratified, random_state=42) lalu skor data uji.

    Pembagiannya identik dengan notebook sehingga tidak ada satu pun baris data uji
    yang pernah dilihat model saat pelatihan.
    """
    X = siapkan_X(_df)
    y = _df["y_bin"]
    idx_train, idx_test = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)

    df_test = _df.iloc[idx_test]
    proba = muat_model().predict_proba(X.iloc[idx_test])[:, 1]
    y_test = y.iloc[idx_test].to_numpy()

    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)

    return {
        "y_test": y_test,
        "proba": proba,
        "skor_manual": df_test["skor_manual"].to_numpy(),
        "n": len(y_test),
        "n_deposan": int(y_test.sum()),
        "base_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc": (fpr, tpr),
        "pr": (rec, prec),
    }


def metrik_pada_ambang(y_test: np.ndarray, proba: np.ndarray, ambang: float) -> dict:
    pred = (proba >= ambang).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "ambang": ambang,
        "f2": float(fbeta_score(y_test, pred, beta=2, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "dihubungi": int(tp + fp),
    }


@st.cache_data(show_spinner=False)
def kurva_f2(y_test: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """F2 di sepanjang grid ambang — grid yang sama dengan notebook."""
    grid = np.arange(0.01, 0.91, 0.01)
    return pd.DataFrame({
        "ambang": grid,
        "f2": [fbeta_score(y_test, (proba >= t).astype(int), beta=2, zero_division=0)
               for t in grid],
    })


@st.cache_data(show_spinner=False)
def tabel_ekonomi(y_test: np.ndarray, proba: np.ndarray,
                  biaya: float, nilai: float) -> pd.DataFrame:
    """Nilai bersih tiap ambang: (TP x nilai deposan) - (jumlah dihubungi x biaya)."""
    baris = []
    for t in np.arange(0.01, 0.91, 0.01):
        m = metrik_pada_ambang(y_test, proba, float(t))
        baris.append({
            "ambang": round(float(t), 2),
            "dihubungi": m["dihubungi"],
            "TP": m["tp"], "FP": m["fp"], "FN": m["fn"],
            "recall": m["recall"], "precision": m["precision"], "f2": m["f2"],
            "nilai_bersih": m["tp"] * nilai - m["dihubungi"] * biaya,
        })
    return pd.DataFrame(baris)


def kurva_gain(y_true: np.ndarray, skor_urut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative gain: % nasabah dihubungi (urut skor tertinggi) vs % deposan tertangkap."""
    urut = np.argsort(skor_urut)[::-1]
    kum = np.cumsum(y_true[urut]) / max(y_true.sum(), 1) * 100
    persen = np.arange(1, len(skor_urut) + 1) / len(skor_urut) * 100
    return persen, kum


@st.cache_data(show_spinner="Menghitung permutation importance...")
def kepentingan_fitur(_df: pd.DataFrame, ambang: float) -> pd.DataFrame:
    """Penurunan F2 ketika satu kolom diacak, diukur pada data uji."""
    X = siapkan_X(_df)
    y = _df["y_bin"]
    _, idx_test = train_test_split(np.arange(len(X)), test_size=0.2,
                                   stratify=y, random_state=42)
    X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]

    def skorer(est, Xs, ys):
        return fbeta_score(ys, (est.predict_proba(Xs)[:, 1] >= ambang).astype(int),
                           beta=2, zero_division=0)

    hasil = permutation_importance(muat_model(), X_test, y_test, n_repeats=5,
                                   random_state=42, scoring=skorer, n_jobs=1)
    return (pd.DataFrame({"fitur": list(X.columns),
                          "importance": hasil.importances_mean,
                          "std": hasil.importances_std})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
