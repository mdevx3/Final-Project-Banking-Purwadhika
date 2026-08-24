"""Pemuatan model dan penskoran."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import streamlit as st

from . import config as C
from .data import siapkan_X


@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    """Pipeline final: RobustScaler + Ordinal/OneHot encoder + HistGradientBoosting."""
    with open(C.MODEL_PATH, "rb") as f:
        return pickle.load(f)


def skor(df_fitur: pd.DataFrame) -> np.ndarray:
    """Probabilitas nasabah membuka deposito. `df_fitur` sudah lewat buat_fitur()."""
    model = muat_model()
    return model.predict_proba(siapkan_X(df_fitur))[:, 1]


def kurva_gain(y_true: np.ndarray, skor_urut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative gain: % nasabah dihubungi (urut skor tertinggi) vs % deposan tertangkap."""
    urut = np.argsort(skor_urut)[::-1]
    kum = np.cumsum(y_true[urut]) / max(y_true.sum(), 1) * 100
    persen = np.arange(1, len(skor_urut) + 1) / len(skor_urut) * 100
    return persen, kum
