"""Aplikasi Streamlit — Prioritas Panggilan Deposito Berjangka.

Dua fitur saja: prediksi manual satu nasabah dan skoring massal satu file.
Final Project Data Science & Machine Learning, Purwadhika.
Jalankan dengan:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Streamlit sudah menambahkan folder skrip ke sys.path, tetapi baris ini membuat
# aplikasi tetap jalan bila dipanggil lewat cara lain (pytest, python -m, dsb).
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Prioritas Panggilan Deposito",
    page_icon=":material/phone_in_talk:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src import config as C                    # noqa: E402
from src.views import batch, prediksi          # noqa: E402

st.markdown(
    """
    <style>
      [data-testid="stMetricValue"] { font-size: 1.6rem; }
      [data-testid="stMetricLabel"] { opacity: .8; }
      .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1250px; }
      h1 { font-size: 1.9rem !important; }
      h3 { margin-top: .4rem; }
      div[data-testid="stExpander"] details { border-radius: .5rem; }
      /* Judul langkah: nomor besar di kiri supaya urutan pemakaian terbaca sekilas */
      .langkah { font-size: 1.05rem; font-weight: 700; margin: 1.4rem 0 .2rem 0; }
      .langkah span { background: #2E8B57; color: #fff; border-radius: 999px;
                      padding: .05rem .58rem; margin-right: .5rem; font-size: .95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Sidebar — hanya satu pengaturan, ditulis dengan bahasa operasional
# --------------------------------------------------------------------------- #
PRESET_AMBANG = {
    "Longgar — jaring lebih banyak calon": 0.07,
    "Standar — disarankan": C.AMBANG_OPERASI,
    "Ketat — hanya yang paling potensial": 0.25,
    "Atur sendiri": None,
}


def _sidebar():
    with st.sidebar:
        st.markdown("## Prioritas Panggilan Deposito")
        st.caption("Model memperkirakan peluang nasabah membuka deposito "
                   "**sebelum** agen menelepon, lalu mengurutkan siapa yang "
                   "ditelepon lebih dulu.")

        st.divider()
        st.markdown("#### Batas masuk antrean")
        pilihan = st.radio(
            "Seberapa selektif daftar panggilan?",
            list(PRESET_AMBANG),
            index=1,
            help="Nasabah dengan peluang di atas batas ini direkomendasikan masuk "
                 "antrean panggilan hari itu.",
        )
        if PRESET_AMBANG[pilihan] is None:
            ambang = st.slider("Batas peluang", 0.01, 0.90, C.AMBANG_OPERASI, 0.01,
                               format="%.2f")
        else:
            ambang = PRESET_AMBANG[pilihan]

        st.session_state["ambang"] = float(ambang)
        st.info(f"Batas dipakai: **peluang {ambang * 100:.0f}% ke atas**",
                icon=":material/tune:")
        if abs(ambang - C.AMBANG_OPERASI) < 1e-9:
            st.caption("Angka bawaan 0,13 berasal dari cross-validation pada data "
                       "latih — bukan dipatok 0,5.")

        st.divider()
        with st.expander("Cara pakai singkat"):
            st.markdown(
                """
**Prediksi Manual** — untuk mengecek satu nasabah.
Isi formulir, tekan *Hitung peluang*. Nilai bawaan sudah terisi, jadi bisa
langsung ditekan untuk mencoba.

**Skoring Massal** — untuk menyusun daftar panggilan harian.
Unggah CSV berisi daftar nasabah (atau pakai data contoh), aplikasi
mengurutkan dari yang paling berpeluang, lalu daftarnya diunduh untuk dialer.

Ubah **Batas masuk antrean** di atas untuk membuat daftar lebih ketat atau
lebih longgar. Hasil di kedua halaman langsung ikut menyesuaikan.
                """
            )
        st.caption("Data latih: 41.172 kontak telemarketing bank ritel Portugal, "
                   "Mei 2008 – November 2010.")
        st.caption("Final Project Data Science — Purwadhika")


_sidebar()

# `url_path` wajib ditulis eksplisit: kedua halaman memakai fungsi bernama sama
# (`render`), sehingga Streamlit tidak bisa menyimpulkan alamat yang unik sendiri.
navigasi = st.navigation([
    # Halaman default selalu dilayani di "/", jadi tidak diberi url_path sendiri.
    st.Page(prediksi.render, title="Prediksi Manual",
            icon=":material/person_search:", default=True),
    st.Page(batch.render, title="Skoring Massal",
            icon=":material/table_rows:", url_path="skoring-massal"),
])
navigasi.run()
