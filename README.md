# Prioritas Panggilan Deposito Berjangka

Aplikasi Streamlit untuk Final Project Data Science & Machine Learning, Purwadhika.
Menerjemahkan model dari notebook `Final_Project_Purwadhika_MID.ipynb` menjadi alat
bantu operasional penyusunan *call list* harian tim telemarketing bank.

---

## 1. Konteks penggunaan

### Masalah

Bank menjual deposito berjangka lewat telemarketing. Dari 41.172 panggilan pada data
historis, hanya **11,27%** berakhir dengan pembukaan deposito — hampir sembilan dari
sepuluh panggilan menghabiskan waktu agen tanpa hasil. Kapasitas panggilan harian itu
tetap, sehingga ketika daftar disusun tanpa urutan, nasabah berpeluang tinggi bisa jatuh
di nomor urut 8.000 dan tidak pernah tersentuh hari itu.

### Siapa penggunanya

| Peran | Hubungan dengan aplikasi |
|---|---|
| **Tim Data/Analytics** | Pengguna langsung. Menjalankan aplikasi terjadwal tiap pagi, mengunggah daftar nasabah aktif beserta kondisi makro terbaru, mengirim hasilnya ke sistem dialer. |
| **Funding Unit** | Pengguna hasil. Menerima daftar yang sudah terurut dan menelepon dari urutan teratas sampai kapasitas harian habis. Tidak mengoperasikan model. |
| **Pimpinan Funding Unit** | Pengambil keputusan. Menetapkan berapa panggilan yang dialokasikan hari itu lewat slider kapasitas pada halaman Skoring Massal. |

### Skenario penggunaan utama

1. Pagi hari, Tim Data menarik daftar nasabah aktif dari core banking.
2. Daftar diunggah ke halaman **Skoring Massal**, lengkap dengan `euribor3m` berjalan.
3. Aplikasi mengembalikan daftar terurut beserta label prioritas.
4. Pimpinan menetapkan kapasitas hari itu lewat slider pada halaman yang sama.
5. Daftar dikirim ke dialer; agen menelepon dari urutan teratas.
6. Hasil panggilan dicatat, drift dipantau, model dilatih ulang tiap kuartal.

Di luar alur harian itu, halaman **Prediksi Manual** dipakai untuk mengecek satu nasabah
yang masuk di luar daftar — misalnya nasabah walk-in atau rujukan cabang.

### Keputusan yang dibantu

| Keputusan | Dibantu oleh |
|---|---|
| Urutan antrean panggilan harian | Skor probabilitas per nasabah, diurutkan menurun |
| Alokasi kapasitas agen hari ini | Slider kapasitas: perkiraan deposan yang tertangkap pada sekian panggilan |
| Nasabah mana yang ditangani agen senior | Label prioritas dan probabilitas individual |
| Layak-tidaknya satu nasabah di luar daftar ditelepon | Halaman Prediksi Manual |

> **Model ini mengurutkan, bukan mencoret.** Dengan asumsi EUR 1,50 per panggilan dan
> EUR 5.000 per deposan, satu panggilan menutup biayanya pada conversion rate 0,03%.
> Segmen terburuk sekalipun masih berkonversi jauh di atas angka itu, sehingga memangkas
> daftar justru menurunkan nilai bersih. Nilai model terletak pada **urutan** ketika
> kapasitas terbatas.

---

## 2. Struktur halaman

Aplikasi sengaja dibatasi pada dua pekerjaan yang benar-benar dilakukan tim setiap hari.

| Halaman | Isi | Elemen interaktif |
|---|---|---|
| **Prediksi Manual** | Skoring satu nasabah lewat formulir tiga bagian | 16 isian bernilai bawaan, dua tombol contoh, gauge, pembanding historis |
| **Skoring Massal** | Unggah CSV (atau pakai data contoh) → call list terurut → unduh | Unggah file, slider kapasitas, filter prioritas, dua tombol unduh |

Keduanya dituntun oleh langkah bernomor, dan seluruh isian sudah terisi nilai bawaan
sehingga aplikasi bisa dicoba tanpa menyiapkan data lebih dulu.

Satu-satunya pengaturan ada di sidebar: **batas masuk antrean** — tiga pilihan siap pakai
(longgar / standar / ketat) plus opsi mengisi angka sendiri. Batas itu berlaku di kedua
halaman lewat `st.session_state`, dan hasil yang sedang tampil langsung ikut menyesuaikan
begitu batasnya diubah.

---

## 3. Model

Isi `best_model.pkl` adalah `Pipeline` scikit-learn utuh (preprocessing + classifier),
hasil `RandomizedSearchCV` pada notebook. Aplikasi memuatnya apa adanya — tidak ada
pelatihan ulang.

```
Pipeline
├── prep : ColumnTransformer
│   ├── num : RobustScaler          (7 kolom)
│   ├── ord : OrdinalEncoder        (education, urutan eksplisit)
│   └── nom : OneHotEncoder         (10 kolom, drop='first', handle_unknown='ignore')
└── clf  : HistGradientBoostingClassifier
```

18 fitur masuk → 49 kolom setelah encoding.

**Ambang operasi 0,13**, ditetapkan dari cross-validation pada data latih (bukan dicari
di data uji, dan bukan dipatok 0,5). Metrik penyeleksi **F2**, karena melewatkan deposan
jauh lebih mahal daripada satu panggilan sia-sia.

Performa pada 8.235 baris data uji yang tidak pernah dilihat model saat pelatihan:

| Metrik | Nilai |
|---|---|
| F2 (ambang 0,13) | 0,5844 |
| Recall | 0,6498 |
| Precision | 0,4167 |
| PR-AUC | 0,4805 |
| ROC-AUC | 0,8169 |
| Gain 10% teratas | 46,7% deposan (lift 4,67x) |

Angka pada tabel di atas berasal dari notebook. Aplikasi sendiri tidak menghitung ulang
metrik evaluasi — ia hanya memuat `best_model.pkl` apa adanya dan memakainya untuk
menskor. Kualitas urutan tetap bisa dilihat langsung di halaman Skoring Massal lewat
kurva cumulative gain pada data contoh, yang dihitung saat itu juga.

### Fitur yang sengaja dibuang

| Kolom | Alasan |
|---|---|
| `duration` | Kebocoran data — nilainya baru ada setelah panggilan selesai, sedangkan model bekerja sebelum panggilan |
| `emp_var_rate`, `cons_price_idx`, `nr_employed` | Korelasi antar fitur makro 0,97; cukup satu wakil (`euribor3m`) |
| `age` mentah | Hubungan berbentuk U, bukan monotonik; diganti `age_group` |
| `pdays` mentah | Digantikan `pdays_clean` + `was_contacted_before` |

---

## 4. Menjalankan secara lokal

Butuh **Python 3.12** atau 3.13 (scikit-learn 1.6.1 belum menyediakan wheel untuk 3.14).

```bash
cd app
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi terbuka di `http://localhost:8501`.

---

## 5. Deploy ke Streamlit Community Cloud

1. Push folder `app/` ini ke sebuah repository GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Isi: repository, branch, dan **Main file path** = `app.py`
   (atau `app/app.py` bila folder ini bukan akar repo).
4. Pada **Advanced settings**, pilih **Python 3.12**.
5. Deploy.

`requirements.txt` mengunci `scikit-learn==1.6.1`, versi yang dipakai melatih
`best_model.pkl`. **Jangan naikkan versinya tanpa melatih ulang model** — scikit-learn
1.8 sudah tidak bisa membaca pickle ini (`_RemainderColsList` dihapus).

---

## 6. Struktur berkas

```
app/
├── app.py                          # entry point, sidebar, navigasi 2 halaman
├── best_model.pkl                  # pipeline final dari notebook
├── requirements.txt                # versi dikunci agar pickle tetap terbaca
├── runtime.txt                     # python-3.12
├── .streamlit/config.toml          # tema & batas ukuran unggahan
├── data/
│   ├── bank_marketing_clean.csv.gz # 41.172 baris hasil cleaning Section C
│   └── template_call_list.csv      # template unggahan, 50 baris contoh
└── src/
    ├── config.py                   # kontrak fitur, ambang, asumsi ekonomi, label
    ├── data.py                     # pemuatan data, rekayasa fitur, validasi unggahan
    ├── model.py                    # pemuatan model, skoring, kurva gain
    ├── charts.py                   # grafik Plotly (gauge & cumulative gain)
    └── views/
        ├── prediksi.py             # halaman Prediksi Manual
        └── batch.py                # halaman Skoring Massal
```

Rekayasa fitur di `src/data.py` mereplikasi Section C dan F.1 notebook persis, dan sudah
diverifikasi menghasilkan metrik data uji yang sama. Nilai kategori pada file unggahan
diseragamkan ke huruf kecil dan kolom angka dipaksa numerik di tahap ini, supaya file
yang menulis `Cellular` atau menyisipkan sel teks pada kolom `age` tetap terskor
(dengan catatan peringatan), bukan menggagalkan proses.

---

## 7. Batasan

- **Ketergantungan pada satu variabel makro.** `euribor3m` menyumbang porsi terbesar
  permutation importance. Bila rezim suku bunga bergeser jauh dari rentang data latih
  (0,63%–5,05%), performa berpotensi turun tajam.
- **Data latih dari kampanye Mei 2008 – November 2010** di sebuah bank ritel Portugal.
- **Kolom `month` hanya memuat 10 bulan** — data tidak berisi satu pun kontak pada
  Januari dan Februari, jadi kedua bulan itu tidak ditawarkan di formulir.
- **Kolom `month` tidak menyertakan tahun**, sehingga efek "bulan emas" bisa jadi hanya
  penanda periode ekonomi tertentu, bukan musiman yang berulang.
- **Keluaran model adalah peringkat, bukan keputusan final.** Kepatuhan, daftar
  jangan-hubungi, dan kebijakan bank tetap berlaku di atasnya.
- **Deep link ke `/skoring-massal` pada permintaan pertama setelah server dingin** bisa
  memunculkan notifikasi "Page not found" sekejap dari Streamlit sebelum navigasi
  terdaftar. Halaman tetap tampil benar dan muat berikutnya bersih; alur normal
  (masuk lewat `/` lalu klik menu) tidak pernah mengalaminya.

## 8. Sumber data

UCI Machine Learning Repository — *Bank Marketing Data Set* (`bank-additional-full.csv`).
S. Moro, P. Cortez, P. Rita (2014). *A Data-Driven Approach to Predict the Success of
Bank Telemarketing*. Decision Support Systems 62:22–31.
