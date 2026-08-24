"""Aplikasi Streamlit — Prioritas Panggilan Deposito Berjangka.

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
from src.views import batch, beranda, insight, model_info, prediksi, simulasi  # noqa: E402

st.markdown(
    """
    <style>
      [data-testid="stMetricValue"] { font-size: 1.65rem; }
      [data-testid="stMetricLabel"] { opacity: .75; }
      .block-container { padding-top: 2.6rem; padding-bottom: 3rem; }
      h1 { font-size: 2rem !important; }
      h3 { margin-top: .6rem; }
      div[data-testid="stExpander"] details { border-radius: .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "ambang" not in st.session_state:
    st.session_state["ambang"] = C.AMBANG_OPERASI


def _sidebar():
    with st.sidebar:
        st.markdown("### Titik kerja model")
        st.slider(
            "Ambang probabilitas", 0.01, 0.90, key="ambang", step=0.01,
            help="Nasabah dengan peluang di atas ambang ini masuk antrean panggilan. "
                 "Nilai bawaan 0,13 ditetapkan dari cross-validation pada data latih, "
                 "bukan dipatok 0,5.",
        )
        if abs(st.session_state["ambang"] - C.AMBANG_OPERASI) > 1e-9:
            st.caption(f"Diubah dari ambang operasi resmi {C.AMBANG_OPERASI:.2f}.")
            if st.button("Kembalikan ke 0,13", use_container_width=True):
                st.session_state["ambang"] = C.AMBANG_OPERASI
                st.rerun()
        else:
            st.caption("Ambang operasi resmi hasil cross-validation.")

        st.divider()
        st.markdown(
            """
            **Prioritas Panggilan Deposito Berjangka**

            Model: HistGradientBoosting pada 18 fitur yang seluruhnya tersedia
            *sebelum* panggilan dilakukan.

            Data: 41.172 kontak telemarketing bank ritel Portugal,
            Mei 2008 – November 2010.
            """
        )
        st.caption("Final Project Data Science — Purwadhika")


_sidebar()

# `url_path` wajib ditulis eksplisit: keenam halaman memakai fungsi bernama sama
# (`render`), sehingga Streamlit tidak bisa menyimpulkan alamat yang unik sendiri.
navigasi = st.navigation([
    # Halaman default selalu dilayani di "/", jadi tidak diberi url_path sendiri.
    st.Page(beranda.render, title="Beranda", icon=":material/home:", default=True),
    st.Page(prediksi.render, title="Prediksi Individual",
            icon=":material/person_search:", url_path="prediksi"),
    st.Page(batch.render, title="Skoring Massal",
            icon=":material/table_rows:", url_path="skoring-massal"),
    st.Page(simulasi.render, title="Simulasi & Ekonomi",
            icon=":material/calculate:", url_path="simulasi"),
    st.Page(insight.render, title="Insight Kampanye",
            icon=":material/insights:", url_path="insight"),
    st.Page(model_info.render, title="Tentang Model",
            icon=":material/model_training:", url_path="tentang-model"),
])
navigasi.run()
