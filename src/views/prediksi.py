"""Halaman 1 — prediksi manual untuk satu nasabah."""
import pandas as pd
import streamlit as st

from .. import charts, config as C
from ..data import buat_fitur, muat_data, periksa_konsistensi
from ..model import skor

# Nilai bawaan = nasabah dengan profil paling umum pada data latih. Formulir sengaja
# terisi penuh sejak awal supaya pengguna baru bisa langsung menekan tombol hitung.
BAWAAN = {
    "p_age": 41, "p_job": "admin.", "p_marital": "married",
    "p_edu": "university.degree", "p_default": "no", "p_housing": "no", "p_loan": "no",
    "p_contact": "cellular", "p_month": "may", "p_day": "mon", "p_campaign": 1,
    "p_poutcome": "nonexistent", "p_previous": 0, "p_pdays": C.PDAYS_BELUM_PERNAH,
    "p_euribor": 1.30, "p_conf": -40.0,
}


def _formulir(punya_hasil: bool):
    """Gambar formulir. Kembalikan dict input mentah bila tombol hitung ditekan.

    Begitu satu hasil sudah dihitung, formulir dilipat ke dalam expander supaya
    hasilnya tidak terdorong jauh ke bawah layar dan pengguna tidak perlu
    menggulir untuk melihat angka yang baru saja dimintanya.
    """
    for kunci, isi in BAWAAN.items():
        st.session_state.setdefault(kunci, isi)

    st.markdown('<div class="langkah"><span>1</span>Isi data nasabah</div>',
                unsafe_allow_html=True)
    if punya_hasil:
        kotak = st.expander("Buka formulir untuk mengubah data nasabah", expanded=False)
    else:
        kotak = st.container()
    with kotak:
        return _isi_formulir()


def _isi_formulir():
    st.caption("Semua kolom sudah terisi nilai bawaan. Ubah yang perlu saja.")

    with st.form("form_nasabah", border=True):
        st.markdown("**Profil nasabah**")
        a1, a2, a3, a4 = st.columns(4)
        a1.slider("Usia", 18, 95, key="p_age")
        a2.selectbox("Pekerjaan", list(C.LABEL_JOB), key="p_job",
                     format_func=lambda v: C.LABEL_JOB.get(v, v))
        a3.selectbox("Status pernikahan", list(C.LABEL_MARITAL), key="p_marital",
                     format_func=lambda v: C.LABEL_MARITAL.get(v, v))
        a4.selectbox("Pendidikan terakhir", C.URUTAN_EDUKASI, key="p_edu",
                     format_func=lambda v: C.LABEL_EDUKASI.get(v, v))

        st.markdown("**Rencana panggilan**")
        b1, b2, b3, b4 = st.columns(4)
        b1.selectbox("Ditelepon lewat", ["cellular", "telephone"], key="p_contact",
                     format_func=lambda v: C.LABEL_CONTACT.get(v, v),
                     help="Nasabah yang dihubungi lewat ponsel berkonversi 14,7%, "
                          "lewat telepon rumah hanya 5,2%.")
        b2.selectbox("Bulan rencana menelepon", C.URUT_BULAN, key="p_month",
                     format_func=lambda v: C.LABEL_BULAN.get(v, v))
        b3.selectbox("Hari rencana menelepon", C.URUT_HARI, key="p_day",
                     format_func=lambda v: C.LABEL_HARI.get(v, v))
        b4.number_input("Panggilan ke- pada kampanye ini", 1, 60, key="p_campaign",
                        help="Peluang berhasil turun konsisten setelah panggilan ke-4.")

        st.markdown("**Riwayat kampanye sebelumnya**")
        c1, c2 = st.columns(2)
        c1.selectbox("Hasil kampanye sebelumnya", list(C.LABEL_POUTCOME), key="p_poutcome",
                     format_func=lambda v: C.LABEL_POUTCOME.get(v, v),
                     help="Petunjuk terkuat yang sudah tersedia sebelum panggilan.")
        c2.number_input("Berapa kali dihubungi pada kampanye sebelumnya", 0, 10,
                        key="p_previous")
        # Satu widget, bukan toggle + slider. Di dalam `st.form` perubahan widget tidak
        # memicu rerun, sehingga `disabled=` yang bergantung pada widget lain akan
        # memakai nilai lama dan slidernya tetap terkunci sampai form disubmit.
        st.select_slider(
            "Berapa hari lalu terakhir dihubungi", options=C.OPSI_PDAYS, key="p_pdays",
            help="Posisi paling kiri berarti nasabah belum pernah dihubungi pada "
                 "kampanye sebelumnya. Geser ke kanan untuk mengisi jarak harinya "
                 "(pada data latih 0–27 hari).")

        st.markdown("**Kondisi ekonomi saat panggilan direncanakan**")
        st.caption("Dua angka ini berlaku untuk seluruh nasabah pada bulan berjalan. "
                   "Kalau belum tahu angka terbarunya, biarkan apa adanya.")
        d1, d2 = st.columns(2)
        d1.slider("Suku bunga Euribor 3 bulan (%)", 0.60, 5.10, step=0.01, key="p_euribor",
                  help="Faktor paling menentukan. Saat suku bunga di bawah 1,5%, "
                       "minat nasabah pada deposito melonjak.")
        d2.slider("Indeks kepercayaan konsumen", -51.0, -26.0, step=0.1, key="p_conf",
                  help="Indikator bulanan; makin mendekati nol makin optimistis konsumen.")

        # Popover, bukan expander: formulir ini sendiri sudah berada di dalam
        # expander begitu satu hasil dihitung, dan expander tidak boleh bersarang.
        with st.popover("Data kredit nasabah (opsional — jarang mengubah hasil)",
                        use_container_width=True):
            e1, e2, e3 = st.columns(3)
            e1.selectbox("Pernah kredit macet", ["no", "unknown", "yes"], key="p_default",
                         format_func=lambda v: C.LABEL_YNU.get(v, v))
            e2.selectbox("Punya KPR", ["no", "unknown", "yes"], key="p_housing",
                         format_func=lambda v: C.LABEL_YNU.get(v, v))
            e3.selectbox("Punya pinjaman pribadi", ["no", "unknown", "yes"], key="p_loan",
                         format_func=lambda v: C.LABEL_YNU.get(v, v))

        kirim = st.form_submit_button("Hitung peluang", type="primary",
                                      use_container_width=True,
                                      icon=":material/calculate:")

    if not kirim:
        return None

    pilih_pdays = st.session_state["p_pdays"]
    return {
        "age": st.session_state["p_age"], "job": st.session_state["p_job"],
        "marital": st.session_state["p_marital"], "education": st.session_state["p_edu"],
        "default": st.session_state["p_default"], "housing": st.session_state["p_housing"],
        "loan": st.session_state["p_loan"], "contact": st.session_state["p_contact"],
        "month": st.session_state["p_month"], "day_of_week": st.session_state["p_day"],
        "campaign": st.session_state["p_campaign"],
        "pdays": 999 if pilih_pdays == C.PDAYS_BELUM_PERNAH else int(pilih_pdays),
        "previous": st.session_state["p_previous"],
        "poutcome": st.session_state["p_poutcome"],
        "cons_conf_idx": st.session_state["p_conf"],
        "euribor3m": st.session_state["p_euribor"],
    }


def _hasil(inp: dict, ambang: float):
    baris = pd.DataFrame([inp])
    masalah = periksa_konsistensi(inp["poutcome"], inp["pdays"], inp["previous"])
    if masalah:
        st.warning(
            "**Kombinasi riwayat kampanye ini tidak pernah muncul pada data latih.** "
            "Peluang tetap dihitung, tetapi angkanya kurang bisa dipercaya.\n\n"
            + "\n".join(f"- {m}" for m in masalah), icon=":material/rule:")

    fitur = buat_fitur(baris)
    prob = float(skor(fitur)[0])

    st.markdown('<div class="langkah"><span>2</span>Hasil</div>', unsafe_allow_html=True)

    # Ketika selisihnya lebih kecil dari pembulatan satu desimal, kedua angka
    # tampil sama di layar. Kalimatnya diganti supaya tidak terbaca menyangkal diri.
    nyaris = abs(prob - ambang) < 0.005
    if prob >= ambang:
        posisi = ("tepat di batas" if nyaris
                  else f"di atas batas {C.desimal(ambang * 100, 0)}%")
        st.success(
            f"**TELEPON NASABAH INI — masukkan ke antrean.**\n\n"
            f"Peluang membuka deposito **{C.desimal(prob * 100)}%**, {posisi} "
            "yang dipilih.", icon=":material/call:")
    else:
        posisi = ("hanya terpaut tipis di bawah batas" if nyaris
                  else f"di bawah batas {C.desimal(ambang * 100, 0)}%")
        st.warning(
            f"**PRIORITAS RENDAH — telepon bila kapasitas masih sisa.**\n\n"
            f"Peluang membuka deposito **{C.desimal(prob * 100)}%**, {posisi} "
            "yang dipilih. Model mengurutkan, bukan mencoret: nasabah ini tetap "
            "boleh ditelepon setelah antrean utama habis.", icon=":material/schedule:")

    kiri, kanan = st.columns([1, 1.25])
    with kiri:
        st.plotly_chart(charts.gauge_probabilitas(prob, ambang), use_container_width=True)
        st.caption(f"Garis hitam = batas masuk antrean "
                   f"({C.desimal(ambang * 100, 0)}%).")

    df = muat_data()
    rata_rata = float(df["y_bin"].mean())
    with kanan:
        m1, m2 = st.columns(2)
        m1.metric("Peluang membuka deposito", f"{C.desimal(prob * 100)}%")
        m2.metric("Dibanding nasabah rata-rata", f"{C.desimal(prob / rata_rata)}x",
                  help=f"Rata-rata nasabah pada data historis berpeluang "
                       f"{C.desimal(rata_rata * 100, 2)}%.")

        nilai_harapan = prob * C.NILAI_PER_DEPOSAN - C.BIAYA_PER_PANGGILAN
        st.metric("Perkiraan nilai satu panggilan", f"EUR {C.ribu(nilai_harapan)}",
                  help=f"(peluang × EUR {C.ribu(C.NILAI_PER_DEPOSAN)} dana deposito) − "
                       f"EUR {C.BIAYA_PER_PANGGILAN:,.2f} biaya menelepon.")

        # Pembanding historis dibatasi 30 panggilan supaya angkanya tidak menyesatkan.
        mirip = df[(df["age_group"].astype(str) == str(fitur["age_group"].iloc[0]))
                   & (df["job"] == inp["job"]) & (df["contact"] == inp["contact"])]
        if len(mirip) >= 30:
            st.info(
                f"Pada data historis, dari **{C.ribu(len(mirip))} panggilan** ke nasabah "
                f"sejenis (usia {fitur['age_group'].iloc[0]}, "
                f"{C.LABEL_JOB[inp['job']].lower()}, "
                f"{C.LABEL_CONTACT[inp['contact']].lower()}), "
                f"**{C.desimal(mirip['y_bin'].mean() * 100)}%** berakhir dengan deposito.",
                icon=":material/groups:")
        else:
            st.info(f"Profil sejenis hanya muncul {len(mirip)} kali pada data historis — "
                    "terlalu sedikit untuk dijadikan pembanding.", icon=":material/groups:")

    with st.expander("Rincian teknis — kolom yang dihitung sendiri oleh aplikasi"):
        turunan = pd.DataFrame({
            "Kolom": ["age_group", "pdays_clean", "was_contacted_before",
                      "has_previous_contact"],
            "Nilai": [str(fitur["age_group"].iloc[0]),
                      str(int(fitur["pdays_clean"].iloc[0])),
                      str(int(fitur["was_contacted_before"].iloc[0])),
                      str(int(fitur["has_previous_contact"].iloc[0]))],
            "Aturan": [
                f"Usia {inp['age']} tahun masuk kelompok binning [0,25,35,45,55,65,100]",
                "Kode 999 (belum pernah dihubungi) diganti -1, bukan 0",
                "0 bila pdays = 999, selain itu 1",
                "1 bila jumlah kontak kampanye sebelumnya > 0",
            ],
        })
        st.dataframe(turunan, hide_index=True, use_container_width=True)
        st.caption("Empat kolom ini tidak diminta dari pengguna karena bisa diturunkan "
                   "sendiri dari isian di atas — persis seperti Section F.1 notebook.")


def render():
    st.title("Prediksi Manual — Satu Nasabah")
    st.caption("Perkirakan peluang seorang nasabah membuka deposito sebelum ditelepon.")

    ambang = st.session_state.get("ambang", C.AMBANG_OPERASI)

    inp = _formulir(punya_hasil="hasil" in st.session_state)
    if inp is not None:
        # Digambar ulang sekali supaya formulir langsung terlipat pada perhitungan
        # pertama; tanpa ini hasil baru muncul di bawah formulir yang masih panjang.
        st.session_state["hasil"] = inp
        st.rerun()

    # Hasil disimpan supaya tidak hilang ketika halaman digambar ulang, misalnya
    # saat batas masuk antrean di sidebar diubah — angka verdict ikut menyesuaikan.
    if "hasil" in st.session_state:
        st.divider()
        _hasil(st.session_state["hasil"], ambang)
    else:
        st.info("Tekan **Hitung peluang** untuk melihat hasilnya.",
                icon=":material/info:")
