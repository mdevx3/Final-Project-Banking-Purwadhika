"""Halaman 4 — simulasi ambang, kapasitas, dan nilai ekonomi."""
import numpy as np
import pandas as pd
import streamlit as st

from .. import charts, components, config as C
from ..data import muat_data
from ..model import (evaluasi_data_uji, kurva_f2, kurva_gain, metrik_pada_ambang,
                     tabel_ekonomi)


def render():
    st.title("Simulasi Ambang & Nilai Ekonomi")
    st.caption("Semua angka di halaman ini dihitung pada 20% data uji — "
               "bagian yang tidak pernah dilihat model saat pelatihan.")

    df = muat_data()
    ev = evaluasi_data_uji(df)
    y_test, proba = ev["y_test"], ev["proba"]

    st.markdown("#### Asumsi ekonomi")
    a1, a2, a3 = st.columns([1, 1, 1.4])
    biaya = a1.number_input("Biaya per panggilan (EUR)", 0.1, 100.0,
                            C.BIAYA_PER_PANGGILAN, 0.1,
                            help="Asumsi Business Understanding: 0,05 EUR/menit x 30 menit.")
    nilai = a2.number_input("Nilai per deposan (EUR)", 10.0, 20000.0,
                            C.NILAI_PER_DEPOSAN, 50.0,
                            help="Penempatan minimal per deposito. Catatan penting: ini nominal "
                                 "dana nasabah, bukan pendapatan bank.")
    ambang = st.session_state.get("ambang", C.AMBANG_OPERASI)
    a3.metric("Conversion rate titik impas", f"{biaya / nilai * 100:.4f}%",
              help="Satu panggilan menutup biayanya bila peluang konversinya di atas angka ini.")

    ekonomi = tabel_ekonomi(y_test, proba, biaya, nilai)
    m = metrik_pada_ambang(y_test, proba, ambang)
    nilai_semua = y_test.sum() * nilai - len(y_test) * biaya
    terbaik = ekonomi.loc[ekonomi["nilai_bersih"].idxmax()]

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Ambang & confusion matrix",
                                "Nilai ekonomi per ambang",
                                "Kapasitas terbatas"])

    # ---------------------------------------------------------------- tab 1
    with tab1:
        st.markdown(f"#### Titik kerja pada ambang **{ambang:.2f}**")
        st.caption("Ubah ambang lewat panel di sidebar kiri untuk melihat pengaruhnya "
                   "ke seluruh halaman.")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("F2", f"{m['f2']:.4f}", help="Metrik utama — recall dinilai 2x lebih penting.")
        k2.metric("Recall", f"{m['recall'] * 100:.1f}%",
                  help=f"{m['tp']:,} dari {ev['n_deposan']:,} deposan tertangkap.")
        k3.metric("Precision", f"{m['precision'] * 100:.1f}%",
                  help=f"{m['tp']:,} benar dari {m['dihubungi']:,} nasabah yang ditelepon.")
        k4.metric("Panggilan dipakai", f"{m['dihubungi']:,}",
                  delta=f"{m['dihubungi'] / ev['n'] * 100:.1f}% dari daftar",
                  delta_color="off")

        c1, c2 = st.columns([1, 1])
        with c1:
            components.confusion_matrix(m["tn"], m["fp"], m["fn"], m["tp"])
        c2.plotly_chart(charts.garis_f2(kurva_f2(y_test, proba), ambang),
                        use_container_width=True)

        m05 = metrik_pada_ambang(y_test, proba, 0.5)
        st.info(
            f"**Kenapa ambangnya tidak 0,5?** Pada ambang bawaan 0,50 model hanya menangkap "
            f"**{m05['tp']:,} dari {ev['n_deposan']:,} deposan** ({m05['recall'] * 100:.1f}%) "
            f"dengan F2 {m05['f2']:.4f}. Pada ambang {ambang:.2f} tangkapannya "
            f"**{m['tp']:,} deposan** ({m['recall'] * 100:.1f}%) dengan F2 {m['f2']:.4f}. "
            "Angka 0,5 mengasumsikan biaya salah tebak setara pada kedua arah, padahal "
            "melewatkan deposan jauh lebih mahal daripada satu panggilan sia-sia.",
            icon=":material/lightbulb:")

    # ---------------------------------------------------------------- tab 2
    with tab2:
        st.plotly_chart(
            charts.garis_ekonomi(ekonomi, nilai_semua, ambang, float(terbaik["ambang"])),
            use_container_width=True)

        e1, e2, e3 = st.columns(3)
        e1.metric("Nilai bersih pada ambang terpilih",
                  f"EUR {m['tp'] * nilai - m['dihubungi'] * biaya:,.0f}")
        e2.metric("Nilai bersih tertinggi",
                  f"EUR {terbaik['nilai_bersih']:,.0f}",
                  delta=f"pada ambang {terbaik['ambang']:.2f}")
        e3.metric("Strategi hubungi semua", f"EUR {nilai_semua:,.0f}")

        batas_nilai = (biaya * (ev["n"] - m["dihubungi"])) / max(m["fn"], 1)
        if terbaik["nilai_bersih"] <= nilai_semua * 1.0001:
            st.warning(
                f"Dengan asumsi sekarang, **tidak ada satu pun ambang yang mengalahkan "
                f"menelepon semua orang**. Titik impas hanya {biaya / nilai * 100:.4f}% "
                "sedangkan segmen terburuk pun masih berkonversi jauh di atas itu, sehingga "
                "setiap panggilan yang dipangkas membuang nilai lebih besar daripada biaya "
                "yang dihemat.\n\n"
                f"Penargetan selektif pada ambang {ambang:.2f} baru menang bila nilai per "
                f"deposan ternyata di bawah **EUR {batas_nilai:,.0f}**. Angka ini relevan "
                f"karena EUR {nilai:,.0f} adalah nominal dana yang ditempatkan nasabah, "
                "bukan pendapatan bank — pendapatan bank atas dana pihak ketiga berupa "
                "margin bunga, yang jauh lebih kecil.",
                icon=":material/warning:")
        else:
            st.success(
                f"Pada asumsi ini penargetan selektif **mengungguli** strategi hubungi semua. "
                f"Ambang paling menguntungkan {terbaik['ambang']:.2f} menghasilkan "
                f"EUR {terbaik['nilai_bersih']:,.0f}, selisih "
                f"EUR {terbaik['nilai_bersih'] - nilai_semua:,.0f} di atas menelepon seluruh "
                "daftar. Kesimpulan ini berbalik dari asumsi bawaan karena nilai per deposan "
                "yang Anda masukkan jauh lebih rendah.",
                icon=":material/check_circle:")

        with st.expander("Tabel lengkap nilai bersih per ambang"):
            st.dataframe(
                ekonomi, hide_index=True, use_container_width=True, height=360,
                column_config={
                    "ambang": st.column_config.NumberColumn("Ambang", format="%.2f"),
                    "dihubungi": st.column_config.NumberColumn("Dihubungi", format="%d"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                    "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                    "f2": st.column_config.NumberColumn("F2", format="%.4f"),
                    "nilai_bersih": st.column_config.NumberColumn(
                        "Nilai bersih (EUR)", format="%.0f"),
                })

    # ---------------------------------------------------------------- tab 3
    with tab3:
        st.markdown("#### Ketika kapasitas panggilan memang terbatas")
        st.markdown(
            "Di sinilah nilai model sebenarnya. Biaya panggilan pada ketiga strategi berikut "
            "**persis sama** karena jumlah panggilannya sama. Yang berbeda hanya urutan "
            "siapa yang ditelepon lebih dulu."
        )

        kapasitas = st.slider("Kapasitas panggilan", 200, ev["n"],
                              min(2000, ev["n"]), 100,
                              help=f"Data uji berisi {ev['n']:,} nasabah "
                                   f"dengan {ev['n_deposan']:,} deposan.")

        def tertangkap(skor_urut):
            urut = np.argsort(skor_urut)[::-1]
            return int(y_test[urut][:kapasitas].sum())

        dep_model = tertangkap(proba)
        dep_manual = tertangkap(ev["skor_manual"])
        dep_acak = kapasitas * ev["base_rate"]

        b1, b2, b3 = st.columns(3)
        b1.metric("Model machine learning", f"{dep_model:,} deposan",
                  delta=f"{dep_model / max(dep_acak, 1e-9):.2f}x lipat urutan acak")
        b2.metric("Skor prioritas manual", f"{dep_manual:,} deposan",
                  delta=f"{dep_manual / max(dep_acak, 1e-9):.2f}x lipat urutan acak")
        b3.metric("Tanpa seleksi (acak)", f"{dep_acak:,.0f} deposan")

        st.success(
            f"Dengan {kapasitas:,} panggilan yang sama, mengurutkan pakai model menangkap "
            f"**{dep_model - dep_acak:,.0f} deposan lebih banyak** daripada menelepon tanpa "
            f"urutan, dan **{dep_model - dep_manual:,} lebih banyak** daripada aturan manual "
            f"analis. Pada asumsi EUR {nilai:,.0f} per deposan, selisih terhadap urutan acak "
            f"setara **EUR {(dep_model - dep_acak) * nilai:,.0f}** tanpa menambah satu sen pun "
            "biaya panggilan.",
            icon=":material/trending_up:")

        x, gain_model = kurva_gain(y_test, proba)
        _, gain_manual = kurva_gain(y_test, ev["skor_manual"])
        st.plotly_chart(
            charts.kurva_gain_plot({"Model machine learning": (x, gain_model),
                                    "Skor prioritas manual": (x, gain_manual)},
                                   penanda=[10, 20, 30, 50]),
            use_container_width=True)

        baris = []
        for p in [10, 20, 30, 40, 50, 70, 100]:
            k = max(int(ev["n"] * p / 100), 1)
            gm = gain_model[k - 1]
            gs = gain_manual[k - 1]
            baris.append({
                "% nasabah dihubungi": f"{p}%",
                "Jumlah panggilan": f"{k:,}",
                "Gain model": f"{gm:.2f}%",
                "Gain skor manual": f"{gs:.2f}%",
                "Lift model": f"{gm / p:.2f}x",
            })
        st.dataframe(pd.DataFrame(baris), hide_index=True, use_container_width=True)
