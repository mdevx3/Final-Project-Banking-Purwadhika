# Final Project — Purwadhika
## Optimasi Kampanye Telemarketing Deposito Berjangka
#### Oleh: Miranda, Noor Ichsan Amrullah, Darrell Lokadeva Lim

**Program**: JC Data Science & Machine Learning, Purwadhika
**Notebook utama**: [`Final_Project_Purwadhika_MID.ipynb`](Final_Project_Purwadhika_MID.ipynb)

🔗 **Aplikasi Prediksi (Streamlit)**: https://app-test-risgkxbblvxwpxxvvr6bmh.streamlit.app/skoring-massal

🔗 **Dashboard (Looker Studio)**: https://datastudio.google.com/reporting/b6f3802d-2f7c-487c-81f3-28fbb0e69941

---

## Ringkasan

Bank ritel Portugal menjalankan kampanye telemarketing deposito berjangka (data historis
Mei 2008–November 2010). Dari **41.172 nasabah** yang dihubungi (setelah cleaning), hanya
**4.639 (11,27%)** yang akhirnya membuka deposito, yang artinya rata-rata dibutuhkan **~22,8
panggilan untuk mendapatkan 1 deposan**, dengan estimasi biaya kampanye ~61.764 Euro.

**Problem statement**: nasabah seperti apa yang berpotensi membuka deposito, agar biaya
telemarketing dapat dialokasikan lebih efisien? Pertanyaan ini diuraikan dengan kerangka
5W-1H (WHO/WHEN/WHY/HOW/WHAT), didukung uji statistik (Z-test proporsi, Mann-Whitney U,
Cramér's V) di setiap temuan, lalu dijawab dua kali: menggunakan skor prioritas berbasis
aturan manual, dan model machine learning supaya keduanya bisa dibandingkan
secara adil sebagai baseline dan improvement.

## Data

- **Sumber**: UCI Machine Learning Repository — *Bank Marketing Data Set*
  (`bank-additional-full.csv`, 41.188 baris x 21 kolom mentah).
- **Cleaning**: hapus 12 baris duplikat penuh dan 4 baris `duration = 0` (panggilan tidak
  tersambung, target `y`-nya otomatis "no" sehingga mengotori conversion rate); `"unknown"`
  pada 6 kolom kategorikal dipertahankan sebagai kategori tersendiri (bukan dihapus/diimputasi)
  karena terbukti membawa sinyal, bukan sekadar missing value; sentinel `pdays = 999`
  ("belum pernah dihubungi") diterjemahkan jadi `-1` + kolom indikator terpisah.
- **Target**: `y` (deposito atau tidak), sangat imbalanced — 88,73% "no" vs 11,27% "yes".

## Temuan Kunci (Exploratory Data Analysis)

| Dimensi | Temuan | Signifikansi |
|---|---|---|
| Pekerjaan | Pelajar (31,4%) & pensiunan (25,3%) jauh di atas blue-collar (6,9%) | z = 28,99; p < 0,001 |
| Usia | Pola **U**, bukan linear — usia >60 (46,9%) dan <25 (21,0%) tertinggi, usia 35–54 titik terendah (~8,5%) | Mann-Whitney signifikan, tapi efek monotonik nyaris nol → usia wajib di-*bin*, bukan dipakai mentah |
| Kanal kontak | Seluler 14,74% vs telepon rumah 5,23% | z = 29,38; p < 0,001 |
| Intensitas panggilan | Konversi turun konsisten dari 13,0% (panggilan ke-1) ke 3,1% (ke-11+) | p < 0,001 (satu arah) |
| Riwayat kampanye (`poutcome`) | `success` 65,1% (lift 5,8x) — prediktor tunggal terkuat (Cramér's V 0,320) | — |
| Indikator makro (`euribor3m`) | Berkorelasi kuat dengan konversi, tapi ternyata **proksi rezim waktu** (2008 krisis vs 2009–10 pemulihan), bukan hubungan dosis-respons murni — di dalam satu rezim korelasinya nol | lihat catatan metodologis |

## Skor Prioritas, Cumulative Gain & Analisis Ekonomi

Sebagai baseline sebelum machine learning, dibangun **skor prioritas berbasis aturan**
(kombinasi `poutcome`, bulan, kelompok usia, kanal, pekerjaan, status kredit) yang membagi
populasi ke 5 segmen (A–E). Kurva cumulative gain menunjukkan **30% nasabah berskor
tertinggi sudah menangkap 61% deposan**.

Analisis titik impas (asumsi: biaya 1,5 EUR/panggilan, nilai minimal 5.000 EUR/deposito)
menghasilkan break-even conversion rate hanya **0,03%** — jauh di bawah conversion rate
segmen terburuk sekalipun (4,21%). Kesimpulannya: **tidak ada satu segmen pun yang layak
dicoret dari daftar panggilan**; skor prioritas bernilai sebagai *urutan antrean dialer*
pada kapasitas terbatas, bukan sebagai filter pemangkas daftar.

## Model Machine Learning

- **Model final**: HistGradientBoosting (hasil hyperparameter tuning), dipilih via 5-fold
  cross-validation memakai metrik **F2** (recall dibobot lebih tinggi, sesuai problem
  imbalance) pada ambang optimal (0,11).
- **Performa data uji**: F2 = 0,5780 · PR-AUC = 0,4722 · ROC-AUC = 0,8133.
- **Fitur dibuang dari model**: `duration` (data leakage — baru diketahui setelah panggilan
  selesai), `emp_var_rate`/`cons_price_idx`/`nr_employed` (multikolinearitas ekstrem dengan
  `euribor3m`), `age` mentah (diganti `age_group`), `pdays` mentah (diganti `pdays_clean` +
  indikator), `quarter` (redundan dengan `month`).
- **Feature importance**: didominasi `euribor3m` (~79% dari total importance positif) —
  temuan ini perlu dibaca bersama catatan metodologis di bawah, karena variabel ini
  sebagian besar menangkap perbedaan rezim waktu, bukan murni efek suku bunga.
- **Hasil bisnis**: pada 30% nasabah teratas, model menangkap **689 deposan** — dibanding
  579 dari skor prioritas manual dan ~278 bila menelepon acak (**lift 2,49x** vs acak).

## Cara Menjalankan

```bash
jupyter notebook Final_Project_Purwadhika_MID.ipynb
```

Aplikasi prediksi interaktif (input data nasabah → probabilitas & rekomendasi kontak)
sudah di-deploy dan bisa diakses langsung tanpa instalasi lokal di link Streamlit di atas.

## Sumber Data & Referensi
*   Moro, et al. 2014. A data-driven approach to predict the success of bank telemarketing. Decision Support System : Portugal.
*   Nechita, et al. 2024. Determinants of Accessing A Term Deposit In A Marketing Campaign - Analysis on A Marketing Campaign at A Portuguese Bank. Revista Economica : Portugal.

