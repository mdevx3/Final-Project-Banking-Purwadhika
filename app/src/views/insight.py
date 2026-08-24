"""Halaman 6 — eksplorasi temuan kampanye secara interaktif."""
import numpy as np
import pandas as pd
import streamlit as st

from .. import charts, config as C
from ..data import muat_data

DIMENSI = {
    "age_group": ("Kelompok usia", C.LABEL_USIA),
    "job": ("Pekerjaan", None),
    "marital": ("Status pernikahan", None),
    "education": ("Pendidikan", C.URUTAN_EDUKASI),
    "default": ("Status kredit macet", ["no", "unknown", "yes"]),
    "housing": ("Kepemilikan KPR", ["no", "unknown", "yes"]),
    "loan": ("Pinjaman pribadi", ["no", "unknown", "yes"]),
    "contact": ("Kanal kontak", ["cellular", "telephone"]),
    "month": ("Bulan kontak", C.URUT_BULAN),
    "day_of_week": ("Hari kontak", C.URUT_HARI),
    "poutcome": ("Hasil kampanye sebelumnya", ["nonexistent", "failure", "success"]),
    "level_euribor": ("Level suku bunga Euribor", ["< 1,5%", "1,5-3%", "3-4%", "4-5,1%"]),
    "urutan_panggilan": ("Panggilan ke-", ["1x", "2x", "3x", "4x", "5x", "6-10x", "11x+"]),
    "segmen_manual": ("Segmen skor prioritas manual", None),
}


@st.cache_data(show_spinner=False)
def _tabel_konversi(df: pd.DataFrame, kolom: str) -> pd.DataFrame:
    g = df.groupby(kolom, observed=True)["y_bin"].agg(volume="count", deposan="sum",
                                                      conv_rate="mean")
    g["conv_rate"] = (g["conv_rate"] * 100).round(2)
    g["pct_volume"] = (g["volume"] / len(df) * 100).round(2)
    g["pct_deposan"] = (g["deposan"] / df["y_bin"].sum() * 100).round(2)
    g["lift"] = (g["conv_rate"] / (df["y_bin"].mean() * 100)).round(2)
    return g.reset_index()


@st.cache_data(show_spinner=False)
def _perkaya(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["level_euribor"] = pd.cut(d["euribor3m"], bins=[0, 1.5, 3, 4, 5.1],
                                labels=["< 1,5%", "1,5-3%", "3-4%", "4-5,1%"])
    d["urutan_panggilan"] = pd.cut(d["campaign"], bins=[0, 1, 2, 3, 4, 5, 10, 100],
                                   labels=["1x", "2x", "3x", "4x", "5x", "6-10x", "11x+"])
    return d


def render():
    st.title("Insight Kampanye")
    st.caption("Temuan dari 41.172 panggilan historis — dasar di balik pilihan fitur model. "
               "Semua grafik bisa diarahkan kursor untuk melihat angka detailnya.")

    df = _perkaya(muat_data())
    baseline = df["y_bin"].mean() * 100

    st.markdown("#### Saring populasi")
    f1, f2, f3 = st.columns(3)
    kanal = f1.multiselect("Kanal kontak", ["cellular", "telephone"],
                           default=["cellular", "telephone"],
                           format_func=lambda v: C.LABEL_CONTACT.get(v, v))
    usia = f2.slider("Rentang usia", 17, 98, (17, 98))
    riwayat = f3.multiselect("Hasil kampanye sebelumnya", list(C.LABEL_POUTCOME),
                             default=list(C.LABEL_POUTCOME),
                             format_func=lambda v: C.LABEL_POUTCOME.get(v, v))

    saring = df[df["contact"].isin(kanal) & df["poutcome"].isin(riwayat)
                & df["age"].between(*usia)]

    if len(saring) < 100:
        st.error("Filter terlalu sempit — sisa data kurang dari 100 baris. "
                 "Longgarkan salah satu filter.", icon=":material/filter_alt_off:")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Panggilan terpilih", f"{len(saring):,}",
              delta=f"{len(saring) / len(df) * 100:.1f}% populasi")
    k2.metric("Deposan", f"{int(saring['y_bin'].sum()):,}")
    k3.metric("Conversion rate", f"{saring['y_bin'].mean() * 100:.2f}%")
    k4.metric("Lift vs seluruh populasi", f"{saring['y_bin'].mean() * 100 / baseline:.2f}x",
              help=f"Baseline seluruh populasi {baseline:.2f}%.")

    st.divider()
    dim = st.selectbox("Bandingkan conversion rate berdasarkan",
                       list(DIMENSI), index=list(DIMENSI).index("month"),
                       format_func=lambda k: DIMENSI.get(k, (k,))[0])

    tabel = _tabel_konversi(saring, dim)
    urutan = DIMENSI[dim][1]
    if urutan:
        tabel[dim] = pd.Categorical(tabel[dim].astype(str), categories=urutan, ordered=True)
        tabel = tabel.sort_values(dim)
    else:
        tabel = tabel.sort_values("conv_rate", ascending=False)

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        charts.bar_konversi(tabel, dim, saring["y_bin"].mean() * 100,
                            f"Conversion Rate per {DIMENSI[dim][0]}", DIMENSI[dim][0]),
        use_container_width=True)
    c2.plotly_chart(
        charts.bar_volume_vs_hasil(tabel, dim, "Beban panggilan vs deposan yang dihasilkan"),
        use_container_width=True)

    tertinggi = tabel.loc[tabel["conv_rate"].idxmax()]
    terendah = tabel.loc[tabel["conv_rate"].idxmin()]
    st.info(
        f"Pada populasi terpilih, **{tertinggi[dim]}** berkonversi paling tinggi "
        f"({tertinggi['conv_rate']:.2f}%, lift {tertinggi['lift']:.2f}x) namun hanya menyerap "
        f"{tertinggi['pct_volume']:.2f}% panggilan. Sebaliknya **{terendah[dim]}** berkonversi "
        f"{terendah['conv_rate']:.2f}% dengan {terendah['pct_volume']:.2f}% panggilan. "
        "Grafik kanan memperlihatkan ketimpangan antara beban panggilan dan hasilnya: batang "
        "biru jauh lebih tinggi dari batang hijau berarti kelompok itu menyerap kapasitas "
        "lebih besar daripada kontribusinya terhadap deposan.",
        icon=":material/insights:")

    with st.expander(f"Tabel angka — {DIMENSI[dim][0]}"):
        tampil = tabel.rename(columns={
            dim: DIMENSI[dim][0], "volume": "Volume panggilan", "deposan": "Deposan",
            "conv_rate": "Conversion rate (%)", "pct_volume": "% dari total panggilan",
            "pct_deposan": "% dari total deposan", "lift": "Lift"})
        st.dataframe(tampil, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### Empat temuan yang paling menentukan")
    t1, t2, t3, t4 = st.tabs(["Suku bunga", "Riwayat kampanye", "Kanal & frekuensi", "Usia"])

    with t1:
        st.markdown(
            """
Ketika Euribor 3 bulan turun di bawah 1,5%, conversion rate melonjak ke sekitar **24%**.
Pada masa suku bunga tinggi konversinya hanya sekitar **4,8%**, padahal periode itu justru
menyerap dua pertiga seluruh panggilan — selisihnya hampir lima kali lipat.

Uji Mann-Whitney memperkuatnya: median Euribor saat deposan dihubungi jauh lebih rendah
daripada saat non-deposan dihubungi, dengan rank-biserial sekitar **-0,49**. Itu efek
terkuat di seluruh dataset, jauh melampaui variabel demografis mana pun.

**Artinya**, bulan-bulan emas bukan bulan yang secara ajaib membuat nasabah tertarik,
melainkan bulan yang kebetulan jatuh pada periode suku bunga rendah. Saat suku bunga acuan
rendah, instrumen berisiko rendah seperti deposito berjangka jadi jauh lebih menarik.

**Rekomendasi:** jadikan suku bunga acuan sebagai pemicu intensitas kampanye, bukan kalender.
            """)
    with t2:
        st.markdown(
            """
Nasabah yang pernah menerima tawaran pada kampanye sebelumnya berkonversi **65,11%**,
atau **5,78x** di atas nasabah yang belum pernah dikontak (8,83%). Selisih ini signifikan
sangat kuat (z = 65,60; p < 0,001) dan menjadikan `poutcome` prediktor terkuat yang
benar-benar tersedia sebelum panggilan dilakukan.

Yang menarik, nasabah yang **pernah menolak** tetap berkonversi **14,23%** — masih di atas
baseline dan lebih tinggi daripada nasabah yang belum pernah disentuh sama sekali.
Penolakan pada kampanye lalu bukan alasan untuk mencoret nasabah.

**Rekomendasi:** jadikan basis data kampanye lama sebagai call list lapis pertama, dengan
urutan `success`, lalu `failure`, lalu `nonexistent`.
            """)
    with t3:
        st.markdown(
            """
Telepon seluler berkonversi **14,74%**, telepon rumah hanya **5,23%** — selisih 9,51 poin
persentase, signifikan sangat kuat (z = 29,38). Meskipun begitu, **36,53% panggilan masih
diarahkan ke telepon rumah**, kanal yang terbukti paling lemah.

Untuk jumlah panggilan, conversion rate turun konsisten: panggilan ke-1 **13,04%**,
ke-3 **10,75%**, ke-5 **7,50%**, dan ke-11 ke atas hanya **3,11%**. Panggilan berulang tidak
membujuk nasabah; nasabah yang berminat cenderung setuju lebih awal.

Dari simulasi aturan berhenti, membatasi maksimal **4 panggilan per nasabah** menghemat
**19,30% beban kerja** dengan risiko kehilangan 6,60% deposan — rasio hemat banding korban
sekitar 2,93 : 1.

**Rekomendasi:** batas keras 4 panggilan per nasabah per kampanye di sistem dialer, dan
prioritaskan nomor seluler.
            """)
    with t4:
        st.markdown(
            """
Hubungan usia dengan konversi berbentuk **U**, bukan garis lurus. Nasabah di atas 60 tahun
berkonversi sekitar **45,5%** dan kelompok di bawah 26 tahun juga tinggi, sedangkan kelompok
35–44 tahun hanya sekitar **8,5%**.

Inilah alasan `age` mentah dibuang dari model dan diganti `age_group`: regresi linear akan
menarik garis lurus melewati lembah tersebut dan menyesatkan. Model berbasis pohon tidak
punya masalah itu karena ia memecah rentang nilai alih-alih menarik garis.

Pola yang sama muncul pada pekerjaan — pelajar **31,4%** dan pensiunan **25,3%** jauh di
atas blue-collar **6,9%** — dan keduanya memang kelompok yang beririsan dengan ujung-ujung
rentang usia tadi.
            """)
