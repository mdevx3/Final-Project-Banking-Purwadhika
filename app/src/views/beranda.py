"""Halaman 1 — konteks bisnis dan konteks penggunaan aplikasi."""
import numpy as np
import streamlit as st

from .. import config as C
from ..data import muat_data
from ..model import evaluasi_data_uji, kurva_gain, metrik_pada_ambang


def render():
    st.title("Prioritas Panggilan Deposito Berjangka")
    st.caption("Alat bantu penyusunan call list harian untuk Funding Unit — "
               "Final Project Data Science, Purwadhika")

    df = muat_data()
    ev = evaluasi_data_uji(df)
    ambang = st.session_state.get("ambang", C.AMBANG_OPERASI)
    m = metrik_pada_ambang(ev["y_test"], ev["proba"], ambang)

    x, gain = kurva_gain(ev["y_test"], ev["proba"])
    gain_10 = gain[int(len(gain) * 0.10)]

    st.markdown("#### Ringkasan")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Conversion rate kampanye", f"{df['y_bin'].mean() * 100:.2f}%",
              help="Dari 41.172 panggilan historis, hanya segini yang berakhir deposito.")
    k2.metric("Deposan tertangkap", f"{m['recall'] * 100:.1f}%",
              help=f"Recall model pada data uji di ambang {ambang:.2f}.")
    k3.metric("Ketepatan panggilan", f"{m['precision'] * 100:.1f}%",
              delta=f"{(m['precision'] / df['y_bin'].mean() - 1) * 100:+.0f}% vs acak",
              help="Precision — dari setiap 100 nasabah yang ditelepon, segini yang jadi deposan.")
    k4.metric("Lift 10% teratas", f"{gain_10 / 10:.2f}x",
              help=f"Menelepon 10% nasabah berskor tertinggi menangkap {gain_10:.1f}% deposan.")

    st.divider()

    kiri, kanan = st.columns([1.15, 1])

    with kiri:
        st.markdown("### Masalah yang diselesaikan")
        st.markdown(
            """
Bank menjual deposito berjangka lewat telemarketing. Dari **41.172 panggilan** pada data
historis, hanya **11,27%** berakhir dengan pembukaan deposito. Artinya hampir sembilan dari
sepuluh panggilan menghabiskan waktu agen tanpa hasil.

Persoalannya bukan sekadar boros. Kapasitas panggilan harian itu tetap — jumlah agen dan jam
kerja tidak bisa dilipatgandakan begitu saja. Ketika daftar panggilan disusun tanpa urutan,
nasabah berpeluang tinggi bisa jatuh di nomor urut 8.000 dan tidak pernah tersentuh hari itu.

Aplikasi ini menjawab satu pertanyaan operasional: **siapa yang ditelepon lebih dulu?**
            """
        )

        st.markdown("### Yang aplikasi ini bantu putuskan")
        st.markdown(
            """
| Keputusan | Dibantu oleh |
|---|---|
| Urutan antrean panggilan harian | Skor probabilitas per nasabah, diurutkan menurun |
| Berapa kapasitas agen dialokasikan hari ini | Simulasi kapasitas: berapa deposan tertangkap per 1.000 panggilan |
| Nasabah mana yang ditangani agen senior | Segmen prioritas A–E dan probabilitas individual |
| Kapan intensitas kampanye dinaikkan | Sensitivitas model terhadap `euribor3m` |
| Kapan berhenti menelepon satu nasabah | Batas keras 4 panggilan per kampanye |
            """
        )

    with kanan:
        st.markdown("### Siapa penggunanya")
        st.markdown(
            """
**Pengguna langsung — Tim Data/Analytics.** Menjalankan aplikasi terjadwal setiap pagi,
mengunggah daftar nasabah aktif beserta kondisi makro terbaru, lalu mengirim hasilnya ke
sistem dialer.

**Pengguna hasil — Funding Unit.** Menerima daftar yang sudah terurut dan menelepon dari
urutan teratas sampai kapasitas harian habis. Mereka tidak mengoperasikan model, hanya
memakai keluarannya.

**Pengambil keputusan — Pimpinan Funding Unit.** Memakai halaman simulasi untuk menetapkan
berapa besar kapasitas yang dialokasikan dan kapan kampanye diintensifkan.
            """
        )

        st.markdown("### Skenario penggunaan utama")
        st.markdown(
            """
1. Pagi hari, Tim Data menarik daftar nasabah aktif dari core banking.
2. Daftar diunggah ke halaman **Skoring Massal**, lengkap dengan `euribor3m` berjalan.
3. Aplikasi mengembalikan daftar terurut beserta segmen prioritas.
4. Pimpinan mengecek halaman **Simulasi** untuk menetapkan kapasitas hari itu.
5. Daftar dikirim ke dialer; agen menelepon dari urutan teratas.
6. Hasil panggilan dicatat, dipantau drift-nya, model dilatih ulang tiap kuartal.
            """
        )

    st.divider()
    st.markdown("### Posisi model pada proses bisnis")
    st.markdown(
        """
Model bekerja **sebelum panggilan dilakukan**, tepat pada tahap penyusunan call list — bukan
saat percakapan berlangsung dan bukan sesudahnya. Posisi inilah yang membuat kolom `duration`
(durasi panggilan) mustahil dipakai sebagai fitur: nilainya baru ada setelah tahap ini lewat.
Memasukkannya akan membuat skor evaluasi terlihat bagus di notebook tapi tidak bisa dipakai
sama sekali di lapangan.
        """
    )

    impas = C.BIAYA_PER_PANGGILAN / C.NILAI_PER_DEPOSAN * 100
    konv_terburuk = (df.groupby("segmen_manual", observed=True)["y_bin"].mean().min() * 100)
    st.warning(
        "**Model ini mengurutkan, bukan mencoret.** Dengan asumsi biaya "
        f"EUR {C.BIAYA_PER_PANGGILAN:,.2f} per panggilan dan EUR {C.NILAI_PER_DEPOSAN:,.0f} "
        f"per deposan, satu panggilan sudah menutup biayanya pada conversion rate "
        f"**{impas:.2f}%**. Segmen terburuk sekalipun masih berkonversi "
        f"**{konv_terburuk:.2f}%**, jauh di atas angka itu, sehingga memangkas daftar justru "
        "menurunkan nilai bersih. Nilai model terletak pada **urutan** ketika kapasitas "
        "terbatas — perhitungan lengkapnya ada di halaman Simulasi & Ekonomi.",
        icon=":material/priority_high:",
    )

    with st.expander("Batasan yang perlu diketahui sebelum memakai keluaran aplikasi ini"):
        st.markdown(
            """
- **Ketergantungan pada satu variabel makro.** `euribor3m` menyumbang porsi terbesar
  kepentingan fitur. Ketika rezim suku bunga bergeser jauh dari rentang data latih
  (0,63%–5,05%), performa model berpotensi turun tajam.
- **Data latih dari kampanye Mei 2008–November 2010** di sebuah bank ritel Portugal.
  Perilaku nasabah pada pasar dan periode lain bisa berbeda.
- **Kolom `month` tidak menyertakan tahun**, sehingga efek "bulan emas" bisa jadi hanya
  penanda periode ekonomi tertentu, bukan musiman yang berulang tiap tahun.
- **Model tidak mengenal nasabah baru tanpa riwayat makro.** Kolom `euribor3m` dan
  `cons_conf_idx` wajib diisi kondisi yang berlaku saat panggilan direncanakan.
- **Keluaran model adalah peringkat, bukan keputusan final.** Pertimbangan kepatuhan,
  daftar jangan-hubungi, dan kebijakan bank tetap berlaku di atasnya.
            """
        )
