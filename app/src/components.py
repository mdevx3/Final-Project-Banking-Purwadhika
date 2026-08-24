"""Komponen tampilan yang dirakit dari elemen Streamlit, bukan dari gambar Plotly.

Confusion matrix sengaja tidak digambar sebagai figure Plotly: teks di dalam figure
berukuran tetap sehingga saling menimpa begitu kolomnya menyempit. Dengan `st.columns`
tata letaknya ikut melebar dan menyempit mengikuti lebar layar.
"""
from __future__ import annotations

import streamlit as st

_KARTU = """
<div style="background:{warna};border-radius:.6rem;padding:.85rem .6rem;
            text-align:center;line-height:1.25;min-height:6.2rem;
            display:flex;flex-direction:column;justify-content:center;">
  <div style="font-size:1.6rem;font-weight:700;">{nilai}</div>
  <div style="font-size:.82rem;font-weight:600;opacity:.85;">{judul}</div>
  <div style="font-size:.72rem;opacity:.6;">{ket}</div>
</div>
"""

_LABEL_BARIS = """
<div style="min-height:6.2rem;display:flex;align-items:center;justify-content:flex-end;
            text-align:right;font-size:.82rem;font-weight:600;opacity:.75;
            padding-right:.5rem;">{teks}</div>
"""


def _kartu(kolom, nilai: int, judul: str, ket: str, warna: str):
    kolom.markdown(_KARTU.format(nilai=f"{nilai:,}", judul=judul, ket=ket, warna=warna),
                   unsafe_allow_html=True)


def confusion_matrix(tn: int, fp: int, fn: int, tp: int):
    """Confusion matrix 2x2 dengan warna sesuai makna tiap sel.

    Warna tidak mengikuti besar angka. Kalau mengikuti, sel true negative yang
    jumlahnya paling besar justru tampil paling pekat padahal paling tidak menarik.
    """
    HIJAU_TUA = "rgba(46,139,87,.30)"
    HIJAU_MUDA = "rgba(46,139,87,.12)"
    KUNING = "rgba(218,165,32,.20)"
    MERAH = "rgba(178,34,34,.18)"

    st.markdown("**Confusion Matrix pada ambang terpilih**")

    h = st.columns([1.15, 2, 2])
    h[1].markdown("<div style='text-align:center;font-size:.78rem;font-weight:600;"
                  "opacity:.7;'>Prediksi: tidak ditelepon</div>", unsafe_allow_html=True)
    h[2].markdown("<div style='text-align:center;font-size:.78rem;font-weight:600;"
                  "opacity:.7;'>Prediksi: ditelepon</div>", unsafe_allow_html=True)

    b1 = st.columns([1.15, 2, 2])
    b1[0].markdown(_LABEL_BARIS.format(teks="Aktual:<br>tidak deposito"),
                   unsafe_allow_html=True)
    _kartu(b1[1], tn, "Benar tidak ditelepon", "kapasitas terhemat", HIJAU_MUDA)
    _kartu(b1[2], fp, "Panggilan terbuang", "ditelepon, menolak", KUNING)

    b2 = st.columns([1.15, 2, 2])
    b2[0].markdown(_LABEL_BARIS.format(teks="Aktual:<br>deposito"), unsafe_allow_html=True)
    _kartu(b2[1], fn, "Deposan terlewat", "kesalahan paling mahal", MERAH)
    _kartu(b2[2], tp, "Deposan tertangkap", "hasil yang dicari", HIJAU_TUA)

    st.caption(f"Dari {tp + fn:,} deposan pada data uji, model menangkap {tp:,} "
               f"({tp / max(tp + fn, 1) * 100:.1f}%) dengan menghabiskan {tp + fp:,} panggilan.")
