"""Pemuatan data dan rekayasa fitur — replika persis Section C & F.1 notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from . import config as C


# --------------------------------------------------------------------------- #
# Rekayasa fitur
# --------------------------------------------------------------------------- #
def buat_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah kolom mentah menjadi 18 fitur yang diminta pipeline model.

    Empat fitur turunan dibuat persis seperti Section F.1 notebook:
      - was_contacted_before : pdays == 999 -> 0, selain itu 1
      - pdays_clean          : sentinel 999 -> -1 (bukan 0, karena 0 = baru dihubungi)
      - has_previous_contact : previous > 0
      - age_group            : binning usia (pola U pada Section D.c)
    """
    d = df.copy()

    # Samakan nama kolom bergaya titik (emp.var.rate) menjadi underscore
    d.columns = [c.replace(".", "_").strip() for c in d.columns]

    pdays = pd.to_numeric(d["pdays"], errors="coerce").fillna(999).astype(int)
    previous = pd.to_numeric(d["previous"], errors="coerce").fillna(0).astype(int)
    age = pd.to_numeric(d["age"], errors="coerce").fillna(0).clip(lower=1, upper=100)

    d["was_contacted_before"] = np.where(pdays == 999, 0, 1)
    d["pdays_clean"] = np.where(pdays == 999, -1, pdays)
    d["has_previous_contact"] = (previous > 0).astype(int)
    d["age_group"] = pd.cut(age, bins=C.BIN_USIA, labels=C.LABEL_USIA)

    for kol in ["campaign", "previous", "euribor3m", "cons_conf_idx"]:
        d[kol] = pd.to_numeric(d[kol], errors="coerce")

    return d


def siapkan_X(df: pd.DataFrame) -> pd.DataFrame:
    """Ambil 18 kolom fitur dengan tipe data yang sama seperti saat model dilatih."""
    X = df[C.FITUR_MODEL].copy()
    # Seluruh kategori pada data latih sudah huruf kecil, jadi input diseragamkan
    # ke bentuk yang sama supaya "Cellular" dan "cellular" tidak jadi dua kategori.
    for kol in C.FITUR_NOMINAL + C.FITUR_ORDINAL:
        X[kol] = X[kol].astype(str).str.strip().str.lower()
    for kol in C.FITUR_NUMERIK:
        X[kol] = pd.to_numeric(X[kol], errors="coerce")
    return X


def skor_prioritas_manual(df: pd.DataFrame) -> pd.Series:
    """Skor aturan manual dari Section D.h — dipakai sebagai pembanding model."""
    s = pd.Series(0.0, index=df.index)
    s += np.where(df["poutcome"] == "success", 3.0,
                  np.where(df["poutcome"] == "failure", 1.0, 0.0))
    s += np.where(df["month"].isin(C.BULAN_EMAS), 2.0,
                  np.where(df["month"] == "apr", 1.0, 0.0))
    s += np.where((df["age"] > 60) | (df["age"] < 26), 1.5, 0.0)
    s += np.where(df["contact"] == "cellular", 1.0, 0.0)
    s += np.where(df["job"].isin(["student", "retired"]), 1.0, 0.0)
    s += np.where(df["default"] == "unknown", -1.0, 0.0)
    return s


def segmen_prioritas(skor: pd.Series) -> pd.Series:
    return pd.cut(skor, bins=[-2, 0.5, 1.5, 2.5, 3.5, 12],
                  labels=["E (terendah)", "D", "C", "B", "A (tertinggi)"])


# --------------------------------------------------------------------------- #
# Pemuatan data historis
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Memuat data kampanye...")
def muat_data() -> pd.DataFrame:
    """Data kampanye historis yang sudah dibersihkan (41.172 baris)."""
    df = pd.read_csv(C.DATA_PATH, compression="gzip")
    df["y_bin"] = (df["y"] == "yes").astype(int)
    df = buat_fitur(df)
    df["skor_manual"] = skor_prioritas_manual(df)
    df["segmen_manual"] = segmen_prioritas(df["skor_manual"])
    return df


@st.cache_data
def muat_template() -> bytes:
    return C.TEMPLATE_PATH.read_bytes()


# --------------------------------------------------------------------------- #
# Validasi file unggahan
# --------------------------------------------------------------------------- #
def validasi_upload(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Kembalikan (error yang menggagalkan, peringatan yang bisa diabaikan)."""
    error, peringatan = [], []

    kolom = {c.replace(".", "_").strip() for c in df.columns}
    hilang = [k for k in C.KOLOM_WAJIB_UPLOAD if k not in kolom]
    if hilang:
        error.append(f"Kolom wajib belum ada: {', '.join(hilang)}")
        return error, peringatan

    if len(df) == 0:
        error.append("File tidak berisi satu baris data pun.")
        return error, peringatan

    d = df.copy()
    d.columns = [c.replace(".", "_").strip() for c in d.columns]

    for kol in ["age", "campaign", "pdays", "previous", "euribor3m", "cons_conf_idx"]:
        n_gagal = pd.to_numeric(d[kol], errors="coerce").isna().sum()
        if n_gagal:
            peringatan.append(f"`{kol}`: {n_gagal:,} nilai bukan angka, diisi nilai default.")

    kategori_sah = {
        "job": list(C.LABEL_JOB), "marital": list(C.LABEL_MARITAL),
        "education": C.URUTAN_EDUKASI, "default": list(C.LABEL_YNU),
        "housing": list(C.LABEL_YNU), "loan": list(C.LABEL_YNU),
        "contact": list(C.LABEL_CONTACT), "month": C.URUT_BULAN,
        "day_of_week": C.URUT_HARI, "poutcome": list(C.LABEL_POUTCOME),
    }
    for kol, sah in kategori_sah.items():
        asing = set(d[kol].astype(str).str.strip().str.lower().unique()) - set(sah)
        if asing:
            contoh = ", ".join(sorted(asing)[:4])
            peringatan.append(
                f"`{kol}`: nilai tak dikenal ({contoh}) diperlakukan sebagai kategori acuan."
            )

    return error, peringatan

def periksa_konsistensi(poutcome: str, pdays: int, previous: int) -> list[str]:
    """Cari kombinasi riwayat kampanye yang tidak pernah muncul pada data latih.

    Ketiga kolom ini saling terikat pada data asli, tanpa satu pun pengecualian dari
    41.172 baris:
      - `poutcome = nonexistent` selalu berpasangan dengan `previous = 0` dan `pdays = 999`
      - `poutcome = success/failure` selalu punya `previous >= 1`
      - `pdays != 999` selalu punya `previous >= 1`

    Kombinasi di luar itu tetap bisa diskor model, tetapi angkanya berada di luar
    sebaran data latih sehingga tidak bisa dipercaya. Karena itu diberi peringatan,
    bukan diblokir.
    """
    pesan = []
    belum_pernah = poutcome == "nonexistent"

    if belum_pernah and pdays != 999:
        pesan.append(
            "`Hasil kampanye sebelumnya` diisi **belum pernah dihubungi**, tetapi "
            "`Jarak hari sejak kontak terakhir` diisi angka. Kalau nasabah memang belum "
            "pernah dihubungi, jaraknya harus **Belum pernah**.")
    if belum_pernah and previous > 0:
        pesan.append(
            "`Hasil kampanye sebelumnya` diisi **belum pernah dihubungi**, tetapi "
            "`Jumlah kontak kampanye sebelumnya` lebih dari 0. Keduanya bertentangan.")
    if not belum_pernah and previous == 0:
        pesan.append(
            "`Hasil kampanye sebelumnya` menyatakan nasabah pernah ditawari, tetapi "
            "`Jumlah kontak kampanye sebelumnya` masih 0. Isi minimal 1.")
    if pdays != 999 and previous == 0:
        pesan.append(
            "`Jarak hari sejak kontak terakhir` diisi angka, tetapi "
            "`Jumlah kontak kampanye sebelumnya` masih 0. Keduanya bertentangan.")
    return pesan
