"""Halaman 5 — performa, isi, dan batasan model."""
import pandas as pd
import streamlit as st

from .. import charts, config as C
from ..data import muat_data
from ..model import (evaluasi_data_uji, info_model, kepentingan_fitur,
                     kurva_gain, metrik_pada_ambang)


def render():
    st.title("Performa & Isi Model")
    st.caption("Seluruh angka dihitung ulang saat aplikasi berjalan, langsung dari "
               "`best_model.pkl` — bukan angka yang ditulis tangan.")

    df = muat_data()
    ev = evaluasi_data_uji(df)
    info = info_model()
    ambang = st.session_state.get("ambang", C.AMBANG_OPERASI)
    m = metrik_pada_ambang(ev["y_test"], ev["proba"], ambang)

    t1, t2, t3 = st.tabs(["Performa data uji", "Kepentingan fitur", "Spesifikasi & batasan"])

    # ---------------------------------------------------------------- tab 1
    with t1:
        st.markdown(f"#### Metrik pada {ev['n']:,} baris data uji "
                    f"({ev['n_deposan']:,} deposan, base rate {ev['base_rate'] * 100:.2f}%)")
        k = st.columns(5)
        k[0].metric("F2", f"{m['f2']:.4f}", help="Metrik utama, ditetapkan sebelum model dilatih.")
        k[1].metric("Recall", f"{m['recall']:.4f}")
        k[2].metric("Precision", f"{m['precision']:.4f}")
        k[3].metric("PR-AUC", f"{ev['pr_auc']:.4f}", help="Metrik pendukung.")
        k[4].metric("ROC-AUC", f"{ev['roc_auc']:.4f}", help="Metrik pendukung.")

        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.kurva_roc(*ev["roc"], ev["roc_auc"]), use_container_width=True)
        c2.plotly_chart(charts.kurva_pr(*ev["pr"], ev["pr_auc"], ev["base_rate"]),
                        use_container_width=True)

        x, gain = kurva_gain(ev["y_test"], ev["proba"])
        _, gain_manual = kurva_gain(ev["y_test"], ev["skor_manual"])
        st.plotly_chart(
            charts.kurva_gain_plot({"Model machine learning": (x, gain),
                                    "Skor prioritas manual": (x, gain_manual)},
                                   penanda=[10, 20, 30, 50]),
            use_container_width=True)

        st.markdown(
            f"""
**Cara membaca ketiganya.** ROC-AUC {ev['roc_auc']:.4f} terlihat tinggi, tetapi sumbu
horizontalnya adalah False Positive Rate yang penyebutnya dibanjiri
{int((ev['y_test'] == 0).sum()):,} nasabah penolak, sehingga kesalahan positif tersamar.
Itu sebabnya ROC-AUC hanya dipakai sebagai pendukung.

PR-AUC {ev['pr_auc']:.4f} terlihat jauh lebih rendah padahal menilai model yang sama.
Bedanya, kurva PR tidak menghitung true negative sama sekali sehingga tidak terangkat oleh
tebakan mudah. Lantainya adalah base rate {ev['base_rate']:.4f}, dan model berada sekitar
{ev['pr_auc'] / ev['base_rate']:.1f}x di atasnya.

Cumulative gain menerjemahkan keduanya ke bahasa operasional: menelepon **10% nasabah
teratas saja sudah menangkap {gain[int(len(gain) * 0.10)]:.1f}% deposan**, lift
{gain[int(len(gain) * 0.10)] / 10:.2f}x.
            """
        )

    # ---------------------------------------------------------------- tab 2
    with t2:
        st.markdown("#### Permutation importance")
        st.caption(f"Nilai satu kolom diacak, lalu diukur seberapa turun F2-nya pada ambang "
                   f"{ambang:.2f}. Semakin besar penurunannya, semakin penting kolom itu. "
                   "Diukur pada data uji, jadi yang dinilai kemampuan generalisasi.")

        imp = kepentingan_fitur(df, ambang)
        st.plotly_chart(charts.bar_kepentingan(imp), use_container_width=True)

        positif = imp.loc[imp["importance"] > 0, "importance"].sum()
        teratas = imp.iloc[0]
        st.error(
            f"**`{teratas['fitur']}` mendominasi.** Kepentingannya {teratas['importance']:.5f}, "
            f"sedangkan fitur terkuat berikutnya `{imp.iloc[1]['fitur']}` hanya "
            f"{imp.iloc[1]['importance']:.5f}. Suku bunga acuan sendirian menyumbang "
            f"**{teratas['importance'] / positif * 100:.1f}%** dari seluruh kepentingan positif. "
            "Ini menegaskan keberhasilan kampanye lebih ditentukan kondisi ekonomi eksternal "
            "daripada karakteristik nasabah — dan sekaligus menjadi risiko terbesar model ini.",
            icon=":material/warning:")

        st.dataframe(imp, hide_index=True, use_container_width=True,
                     column_config={
                         "fitur": st.column_config.TextColumn("Fitur"),
                         "importance": st.column_config.NumberColumn(
                             "Penurunan F2", format="%.5f"),
                         "std": st.column_config.NumberColumn("Simpangan baku", format="%.5f"),
                     })

        st.info(
            "Kecilnya kontribusi `age_group` dan `job` di sini **tidak** bertentangan dengan "
            "temuan EDA bahwa usia dan pekerjaan berasosiasi kuat dengan konversi. Yang diukur "
            "berbeda: EDA mengukur asosiasi satu fitur secara sendirian, sedangkan permutation "
            "importance mengukur tambahan informasi setelah seluruh fitur lain sudah ada. "
            "Pensiunan dan pelajar memang berkonversi tinggi, tetapi begitu model sudah tahu "
            "suku bunga dan bulan, informasi demografis hampir tidak menambah apa-apa.",
            icon=":material/info:")

    # ---------------------------------------------------------------- tab 3
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Algoritma")
            st.markdown(
                f"""
Model final adalah **{info['algoritma']}** hasil `RandomizedSearchCV` 50 kombinasi,
dipilih murni berdasarkan skor cross-validation pada data latih. Data uji baru dibuka
sekali, pada tahap evaluasi.

| Hyperparameter | Nilai |
|---|---|
| learning_rate | {info['hyperparameter']['learning_rate']} |
| max_iter | {info['hyperparameter']['max_iter']} |
| max_leaf_nodes | {info['hyperparameter']['max_leaf_nodes']} |
| min_samples_leaf | {info['hyperparameter']['min_samples_leaf']} |
| l2_regularization | {info['hyperparameter']['l2_regularization']} |

Parameternya cenderung konservatif — langkah belajar kecil dengan regularisasi aktif —
dan itu masuk akal mengingat deposan pada data latih hanya sekitar 3.700 orang, sehingga
daun yang terlalu kecil gampang menghafal noise.
                """)

            st.markdown("#### Metrik penyeleksi")
            st.markdown(
                """
**F2**, ditetapkan sebelum satu pun model dilatih. Accuracy tidak dipakai karena model
yang menjawab "tidak" untuk semua orang langsung benar 88,73% tanpa mempelajari apa pun.
F1 juga tidak dipakai karena menimbang precision dan recall sama berat, padahal ongkos
keduanya jauh berbeda. Dengan β = 2, recall dinilai dua kali lebih penting sehingga
False Negative dihukum lebih berat daripada False Positive.
                """)

        with c2:
            st.markdown("#### Pipeline preprocessing")
            st.markdown(
                f"""
Seluruh transformasi dibungkus dalam satu `Pipeline`, di-*fit* hanya pada data latih agar
statistik data uji tidak bocor ke pelatihan. {info['n_fitur_masuk']} fitur masuk, menjadi
**{info['n_kolom_setelah_encoding']} kolom** setelah encoding.

| Blok | Transformer | Jumlah kolom |
|---|---|---|
| Numerik | `RobustScaler` (median & IQR, tahan outlier) | {len(C.FITUR_NUMERIK)} |
| Ordinal | `OrdinalEncoder` dengan urutan pendidikan eksplisit | {len(C.FITUR_ORDINAL)} |
| Nominal | `OneHotEncoder(drop='first', handle_unknown='ignore')` | {len(C.FITUR_NOMINAL)} |
                """)

            st.markdown("#### Kolom yang sengaja dibuang")
            st.markdown(
                """
| Kolom | Alasan |
|---|---|
| `duration` | Kebocoran data — nilainya baru ada setelah panggilan selesai |
| `emp_var_rate`, `cons_price_idx`, `nr_employed` | Korelasi antar fitur makro 0,97; cukup satu wakil |
| `age` mentah | Hubungannya berbentuk U, bukan monotonik; diganti `age_group` |
| `pdays` mentah | Digantikan `pdays_clean` + `was_contacted_before` |
                """)

        st.markdown("#### Kategori yang dikenali model")
        st.caption("Nilai di luar daftar ini akan diperlakukan sebagai kategori acuan "
                   "(`handle_unknown='ignore'`), sehingga aplikasi hanya menawarkan nilai berikut.")
        kat = pd.DataFrame([
            {"Fitur": k, "Jumlah kategori": len(v), "Daftar": ", ".join(v)}
            for k, v in info["kategori_nominal"].items()
        ])
        st.dataframe(kat, hide_index=True, use_container_width=True)
        st.caption("Perhatikan `month` hanya berisi 10 bulan: data kampanye tidak memuat "
                   "satu pun kontak pada Januari dan Februari.")

        st.markdown("#### Batasan dan rencana pemeliharaan")
        st.markdown(
            f"""
- **Risiko utama: ketergantungan pada satu variabel makro.** Bila rezim suku bunga bergeser
  jauh dari rentang data latih, performa berpotensi turun tajam. Pantau distribusi
  `euribor3m` yang masuk setiap bulan dan bandingkan dengan rentang latih.
- **Latih ulang tiap kuartal**, atau lebih cepat bila terjadi perubahan kebijakan suku bunga
  acuan yang material.
- **Pantau drift** pada distribusi skor keluaran dan pada conversion rate aktual per segmen.
  Bila conversion rate segmen teratas turun mendekati baseline, model perlu ditinjau.
- **Ambang operasi {ambang:.2f}** ditetapkan dari cross-validation pada data latih, bukan
  dicari di data uji. Kalau ambang dicari di data uji, angka evaluasinya tidak lagi jujur.
- **Fitur yang berpotensi menaikkan performa** — riwayat transaksi nasabah, saldo rata-rata,
  kepemilikan produk lain — tidak tersedia pada dataset ini. Tuning hyperparameter tambahan
  hanya menaikkan F2 sepersekian ribu, jadi batasnya ada pada informasi, bukan algoritma.
            """
        )
