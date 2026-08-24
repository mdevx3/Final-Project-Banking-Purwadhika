"""Halaman 2 — skoring satu nasabah lewat formulir."""
import pandas as pd
import streamlit as st

from .. import charts, config as C
from ..data import (buat_fitur, muat_data, periksa_konsistensi, segmen_prioritas,
                    skor_prioritas_manual)
from ..model import skor


def render():
    st.title("Prediksi Nasabah Individual")
    st.caption("Hitung peluang satu nasabah membuka deposito sebelum agen menelepon.")

    df = muat_data()
    ambang = st.session_state.get("ambang", C.AMBANG_OPERASI)

    with st.form("form_nasabah"):
        st.markdown("#### Profil nasabah")
        a1, a2, a3 = st.columns(3)
        age = a1.slider("Usia", 18, 95, 41,
                        help="Dikelompokkan otomatis menjadi `age_group` (pola konversi berbentuk U).")
        job = a2.selectbox("Pekerjaan", list(C.LABEL_JOB),
                           format_func=lambda v: C.LABEL_JOB.get(v, v), index=0)
        marital = a3.selectbox("Status pernikahan", list(C.LABEL_MARITAL),
                               format_func=lambda v: C.LABEL_MARITAL.get(v, v), index=1)
        b1, b2, b3, b4 = st.columns(4)
        education = b1.selectbox("Pendidikan terakhir", C.URUTAN_EDUKASI,
                                 format_func=lambda v: C.LABEL_EDUKASI.get(v, v), index=6)
        default = b2.selectbox("Kredit macet", ["no", "unknown", "yes"],
                               format_func=lambda v: C.LABEL_YNU.get(v, v), index=0)
        housing = b3.selectbox("Punya KPR", ["no", "unknown", "yes"],
                               format_func=lambda v: C.LABEL_YNU.get(v, v), index=0)
        loan = b4.selectbox("Punya pinjaman pribadi", ["no", "unknown", "yes"],
                            format_func=lambda v: C.LABEL_YNU.get(v, v), index=0)

        st.markdown("#### Rencana kontak")
        c1, c2, c3, c4 = st.columns(4)
        contact = c1.selectbox("Kanal kontak", ["cellular", "telephone"],
                               format_func=lambda v: C.LABEL_CONTACT.get(v, v), index=0,
                               help="Seluler berkonversi 14,7% vs telepon rumah 5,2%.")
        month = c2.selectbox("Bulan rencana kontak", C.URUT_BULAN,
                             format_func=lambda v: C.LABEL_BULAN.get(v, v), index=2)
        day_of_week = c3.selectbox("Hari rencana kontak", C.URUT_HARI,
                                   format_func=lambda v: C.LABEL_HARI.get(v, v), index=0)
        campaign = c4.number_input("Panggilan ke- pada kampanye ini", 1, 60, 1,
                                   help="Conversion rate turun konsisten setelah panggilan ke-4.")

        st.markdown("#### Riwayat kampanye sebelumnya")
        d1, d2 = st.columns(2)
        poutcome = d1.selectbox("Hasil kampanye sebelumnya", list(C.LABEL_POUTCOME),
                                format_func=lambda v: C.LABEL_POUTCOME.get(v, v), index=2,
                                help="Prediktor terkuat yang tersedia sebelum panggilan.")
        previous = d2.number_input("Jumlah kontak kampanye sebelumnya", 0, 10, 0)

        # Satu widget, bukan toggle + slider. Di dalam `st.form` perubahan widget tidak
        # memicu rerun, sehingga `disabled=` yang bergantung pada widget lain akan
        # memakai nilai lama dan slidernya tetap terkunci sampai form disubmit.
        pilih_pdays = st.select_slider(
            "Jarak hari sejak kontak terakhir kampanye lalu",
            options=C.OPSI_PDAYS, value=C.PDAYS_BELUM_PERNAH,
            help="Posisi paling kiri berarti nasabah belum pernah dihubungi pada kampanye "
                 "sebelumnya — pada data asli berkode 999. Geser ke kanan untuk mengisi "
                 "jarak harinya (rentang pada data latih 0-27 hari).")
        pdays = 999 if pilih_pdays == C.PDAYS_BELUM_PERNAH else int(pilih_pdays)

        st.markdown("#### Kondisi makroekonomi saat panggilan direncanakan")
        e1, e2 = st.columns(2)
        euribor3m = e1.slider("Euribor 3 bulan (%)", 0.60, 5.10, 1.30, 0.01,
                              help="Fitur paling menentukan. Konversi melonjak saat di bawah 1,5%.")
        cons_conf_idx = e2.slider("Consumer confidence index", -51.0, -26.0, -40.0, 0.1,
                                  help="Indeks kepercayaan konsumen, indikator bulanan.")

        kirim = st.form_submit_button("Hitung peluang", type="primary",
                                      use_container_width=True)

    if not kirim:
        st.info("Isi formulir di atas lalu tekan **Hitung peluang**. "
                "Nilai bawaan mewakili nasabah dengan profil rata-rata.",
                icon=":material/info:")
        return

    baris = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan, "contact": contact,
        "month": month, "day_of_week": day_of_week, "campaign": campaign,
        "pdays": pdays, "previous": previous, "poutcome": poutcome,
        "cons_conf_idx": cons_conf_idx, "euribor3m": euribor3m,
    }])
    masalah = periksa_konsistensi(poutcome, pdays, previous)
    if masalah:
        st.warning(
            "**Kombinasi riwayat kampanye ini tidak pernah muncul pada data latih.** "
            "Skor tetap dihitung, tetapi berada di luar sebaran data sehingga tidak bisa "
            "dipercaya sepenuhnya.\n\n" + "\n".join(f"- {m}" for m in masalah),
            icon=":material/rule:")

    fitur = buat_fitur(baris)
    prob = float(skor(fitur)[0])

    skor_m = float(skor_prioritas_manual(fitur).iloc[0])
    segmen = str(segmen_prioritas(pd.Series([skor_m])).iloc[0])

    st.divider()
    st.markdown("### Hasil")
    kiri, kanan = st.columns([1, 1.4])

    with kiri:
        st.plotly_chart(charts.gauge_probabilitas(prob, ambang), use_container_width=True)
        st.caption(f"Garis hitam adalah ambang operasi {ambang:.2f}.")

    with kanan:
        if prob >= ambang:
            st.success(f"**Masukkan ke antrean panggilan.** Peluang {prob * 100:.2f}% "
                       f"berada di atas ambang operasi {ambang:.2f}.", icon=":material/call:")
        else:
            st.warning(f"**Prioritas rendah.** Peluang {prob * 100:.2f}% berada di bawah "
                       f"ambang operasi {ambang:.2f}. Tetap boleh ditelepon bila kapasitas "
                       "masih tersisa — model mengurutkan, bukan mencoret.",
                       icon=":material/schedule:")

        m1, m2, m3 = st.columns(3)
        m1.metric("Peluang konversi", f"{prob * 100:.2f}%")
        m2.metric("Lift vs rata-rata", f"{prob / df['y_bin'].mean():.2f}x",
                  help="Berapa kali lipat di atas conversion rate populasi (11,27%).")
        m3.metric("Segmen aturan manual", segmen,
                  help=f"Skor aturan {skor_m:+.1f} dari baseline analis (Section D.h notebook). "
                       "Dipakai sebagai pembanding, bukan keluaran model.")

        nilai_harapan = prob * C.NILAI_PER_DEPOSAN - C.BIAYA_PER_PANGGILAN
        st.metric("Nilai harapan satu panggilan",
                  f"EUR {nilai_harapan:,.2f}",
                  help=f"(peluang x EUR {C.NILAI_PER_DEPOSAN:,.0f}) - "
                       f"EUR {C.BIAYA_PER_PANGGILAN:,.2f} biaya panggilan.")

    with st.expander("Fitur turunan yang dihitung aplikasi dari input di atas"):
        turunan = pd.DataFrame({
            "Fitur": ["age_group", "pdays_clean", "was_contacted_before", "has_previous_contact"],
            "Nilai": [str(fitur["age_group"].iloc[0]),
                      str(int(fitur["pdays_clean"].iloc[0])),
                      str(int(fitur["was_contacted_before"].iloc[0])),
                      str(int(fitur["has_previous_contact"].iloc[0]))],
            "Aturan": [
                f"Usia {age} tahun masuk kelompok binning [0,25,35,45,55,65,100]",
                "Kode 999 (belum pernah dihubungi) diganti -1, bukan 0",
                "0 bila pdays = 999, selain itu 1",
                "1 bila jumlah kontak kampanye sebelumnya > 0",
            ],
        })
        st.dataframe(turunan, hide_index=True, use_container_width=True)
        st.caption("Empat fitur ini tidak diminta dari pengguna karena bisa diturunkan "
                   "sendiri — persis seperti Section F.1 notebook.")

    st.markdown("### Perbandingan dengan nasabah serupa pada data historis")
    mirip = df[(df["age_group"].astype(str) == str(fitur["age_group"].iloc[0]))
               & (df["job"] == job) & (df["contact"] == contact)]
    if len(mirip) >= 30:
        st.info(
            f"Pada data historis ada **{len(mirip):,} panggilan** ke nasabah dengan kombinasi "
            f"usia *{fitur['age_group'].iloc[0]}*, pekerjaan *{C.LABEL_JOB[job]}*, kanal "
            f"*{C.LABEL_CONTACT[contact]}*. Sebanyak **{mirip['y_bin'].mean() * 100:.2f}%** "
            "di antaranya berakhir dengan deposito.",
            icon=":material/groups:")
    else:
        st.info(f"Kombinasi profil ini hanya muncul {len(mirip)} kali pada data historis — "
                "terlalu sedikit untuk dijadikan pembanding yang andal.",
                icon=":material/groups:")
