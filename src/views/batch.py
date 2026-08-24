"""Halaman 2 — skoring massal: satu file nasabah menjadi daftar panggilan terurut."""
import io

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts, config as C
from ..data import (buat_fitur, muat_data, muat_template, skor_prioritas_manual,
                    validasi_upload)
from ..model import kurva_gain, skor

# Urutan prioritas ditulis eksplisit supaya filter tidak terurut alfabetis
URUT_PRIORITAS = ["Sangat tinggi", "Tinggi", "Sedang", "Rendah", "Sangat rendah"]

# Nama kolom hasil dalam bahasa manusia, dipakai pada tabel di layar
LABEL_KOLOM = {
    "peringkat": "No.", "customer_id": "ID nasabah", "peluang_persen": "Peluang",
    "prioritas": "Prioritas", "rekomendasi": "Telepon hari ini?",
    "nilai_harapan_eur": "Perkiraan nilai (EUR)", "age": "Usia", "job": "Pekerjaan",
    "contact": "Kanal", "month": "Bulan", "poutcome": "Riwayat kampanye lalu",
    "euribor3m": "Euribor 3 bln",
}
KOLOM_TAMPIL = list(LABEL_KOLOM)

# Kolom berkode bahasa Inggris diterjemahkan hanya untuk tampilan; berkas yang
# diunduh tetap memakai nilai asli supaya bisa langsung dibaca sistem dialer.
TERJEMAHAN = {"job": C.LABEL_JOB, "contact": C.LABEL_CONTACT, "month": C.LABEL_BULAN,
              "poutcome": C.LABEL_POUTCOME, "marital": C.LABEL_MARITAL,
              "education": C.LABEL_EDUKASI, "day_of_week": C.LABEL_HARI}


def _prioritas(p: float, ambang: float) -> str:
    if p >= max(ambang * 3, 0.5):
        return "Sangat tinggi"
    if p >= ambang * 2:
        return "Tinggi"
    if p >= ambang:
        return "Sedang"
    if p >= ambang / 2:
        return "Rendah"
    return "Sangat rendah"


@st.cache_data(show_spinner="Menghitung peluang untuk seluruh baris...")
def _skor_tabel(df_mentah: pd.DataFrame) -> pd.DataFrame:
    """Skoring saja — tanpa ambang, supaya mengubah batas di sidebar tidak
    memicu perhitungan model ulang untuk ribuan baris."""
    fitur = buat_fitur(df_mentah)
    hasil = df_mentah.copy()
    hasil.columns = [c.replace(".", "_").strip() for c in hasil.columns]
    hasil["probabilitas"] = skor(fitur)
    hasil["skor_manual"] = skor_prioritas_manual(fitur).to_numpy()
    hasil = hasil.sort_values("probabilitas", ascending=False).reset_index(drop=True)
    hasil.insert(0, "peringkat", np.arange(1, len(hasil) + 1))
    return hasil


def _lengkapi(hasil: pd.DataFrame, ambang: float) -> pd.DataFrame:
    prob = hasil["probabilitas"]
    hasil = hasil.copy()
    hasil["peluang_persen"] = prob * 100
    hasil["prioritas"] = [_prioritas(p, ambang) for p in prob]
    hasil["rekomendasi"] = np.where(prob >= ambang, "Ya", "Bila kapasitas sisa")
    hasil["nilai_harapan_eur"] = prob * C.NILAI_PER_DEPOSAN - C.BIAYA_PER_PANGGILAN
    return hasil


def _panduan_format():
    st.markdown("File CSV harus memuat **16 kolom** berikut. Kolom tambahan seperti "
                "`customer_id` atau nomor telepon boleh ikut dan akan dibawa ke hasil.")
    spek = pd.DataFrame([
        ("age", "Angka", "18–95", "Usia nasabah"),
        ("job", "Teks", ", ".join(list(C.LABEL_JOB)[:4]) + ", ...", "Jenis pekerjaan"),
        ("marital", "Teks", "divorced / married / single / unknown", "Status pernikahan"),
        ("education", "Teks", "basic.4y ... university.degree / unknown", "Pendidikan terakhir"),
        ("default", "Teks", "no / yes / unknown", "Punya kredit macet"),
        ("housing", "Teks", "no / yes / unknown", "Punya KPR"),
        ("loan", "Teks", "no / yes / unknown", "Punya pinjaman pribadi"),
        ("contact", "Teks", "cellular / telephone", "Kanal kontak"),
        ("month", "Teks", "mar, apr, may, ... dec", "Bulan rencana kontak"),
        ("day_of_week", "Teks", "mon / tue / wed / thu / fri", "Hari rencana kontak"),
        ("campaign", "Angka", ">= 1", "Panggilan ke- pada kampanye ini"),
        ("pdays", "Angka", "0–30, atau 999", "Hari sejak kontak terakhir; 999 = belum pernah"),
        ("previous", "Angka", ">= 0", "Jumlah kontak kampanye sebelumnya"),
        ("poutcome", "Teks", "success / failure / nonexistent", "Hasil kampanye sebelumnya"),
        ("cons_conf_idx", "Angka", "-51 s/d -26", "Indeks kepercayaan konsumen berjalan"),
        ("euribor3m", "Angka", "0,6–5,1", "Suku bunga Euribor 3 bulan berjalan"),
    ], columns=["Kolom", "Tipe", "Nilai yang diterima", "Keterangan"])
    st.dataframe(spek, hide_index=True, use_container_width=True)
    st.caption("Kolom `duration` sengaja tidak diminta: durasi panggilan baru diketahui "
               "setelah panggilan selesai, sedangkan model bekerja sebelum panggilan.")


def _baca_csv(berkas) -> pd.DataFrame | None:
    isi = berkas.getvalue()
    for pemisah in (",", ";"):
        calon = pd.read_csv(io.BytesIO(isi), sep=pemisah)
        if calon.shape[1] > 1:
            return calon
    return None


def _pilih_data() -> pd.DataFrame | None:
    st.markdown('<div class="langkah"><span>1</span>Pilih daftar nasabah</div>',
                unsafe_allow_html=True)
    sumber = st.radio(
        "Sumber data", ["Pakai data contoh (langsung coba)", "Unggah file CSV saya"],
        horizontal=True, label_visibility="collapsed")

    if sumber.startswith("Pakai data contoh"):
        n = st.slider("Jumlah nasabah contoh", 500, 8000, 3000, 500)
        df_pop = muat_data()
        contoh = (df_pop.sample(n, random_state=42)
                  .reset_index(drop=True)[C.KOLOM_WAJIB_UPLOAD + ["y"]])
        contoh.insert(0, "customer_id",
                      [f"CUST-{i:05d}" for i in range(1, len(contoh) + 1)])
        contoh = contoh.rename(columns={"y": "hasil_sebenarnya"})
        st.caption(f"{C.ribu(n)} nasabah diambil acak dari data historis. Kolom "
                   "`hasil_sebenarnya` ikut dibawa hanya untuk mengecek kualitas "
                   "urutan; pada penggunaan nyata kolom ini tentu belum ada.")
        return contoh

    u1, u2 = st.columns([2, 1])
    berkas = u1.file_uploader("Pilih file CSV", type=["csv"],
                              help="Pemisah koma maupun titik koma sama-sama diterima.")
    u2.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    u2.download_button("Unduh template CSV", data=muat_template(),
                       file_name="template_call_list.csv", mime="text/csv",
                       use_container_width=True, icon=":material/download:",
                       help="Berisi 50 baris contoh dengan kolom yang benar.")
    with st.expander("Kolom apa saja yang harus ada di file?"):
        _panduan_format()

    if berkas is None:
        st.info("Belum ada file. Unduh template di atas sebagai contoh isian, atau "
                "pilih **Pakai data contoh** untuk langsung mencoba aplikasi.",
                icon=":material/upload_file:")
        return None

    df_mentah = _baca_csv(berkas)
    if df_mentah is None:
        st.error("File tidak bisa dibaca sebagai CSV, baik dengan pemisah koma maupun "
                 "titik koma.", icon=":material/error:")
        return None

    error, peringatan = validasi_upload(df_mentah)
    if error:
        st.error("File belum bisa diproses:\n\n" + "\n".join(f"- {e}" for e in error),
                 icon=":material/error:")
        return None
    if peringatan:
        with st.expander(f"{len(peringatan)} catatan pada file (tidak menggagalkan proses)"):
            for w in peringatan:
                st.markdown(f"- {w}")
    st.success(f"File terbaca: **{C.ribu(len(df_mentah))} baris**, "
               f"{df_mentah.shape[1]} kolom.",
               icon=":material/check_circle:")
    return df_mentah


def _ringkasan(hasil: pd.DataFrame, ambang: float) -> int:
    st.markdown('<div class="langkah"><span>2</span>Tentukan kapasitas hari ini</div>',
                unsafe_allow_html=True)

    n_antre = int((hasil["probabilitas"] >= ambang).sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Nasabah dinilai", C.ribu(len(hasil)))
    k2.metric(f"Layak ditelepon (peluang ≥ {C.desimal(ambang * 100, 0)}%)",
              C.ribu(n_antre),
              delta=f"{n_antre / len(hasil) * 100:.0f}% dari daftar")
    k3.metric("Peluang tertinggi di daftar", f"{hasil['probabilitas'].max() * 100:.0f}%")

    # Batas bawah slider harus tetap di bawah batas atasnya. File pendek — termasuk
    # template 50 baris milik aplikasi ini sendiri — membuat keduanya bertemu dan
    # Streamlit menolak menggambar slidernya.
    n = len(hasil)
    if n < 2:
        kapasitas = n
        st.caption("Daftar hanya berisi satu nasabah, jadi seluruhnya ditelepon.")
    else:
        batas_bawah = 50 if n > 50 else 1
        kapasitas = st.slider(
            "Berapa nasabah yang sanggup ditelepon hari ini?",
            min_value=batas_bawah, max_value=n,
            value=int(min(max(n_antre, batas_bawah), n)),
            step=max(n // 100, 1),
            help="Aplikasi mengambil sekian nasabah teratas dari daftar terurut.")

    teratas = hasil.head(kapasitas)
    acak = kapasitas * hasil["probabilitas"].mean()
    s1, s2, s3 = st.columns(3)
    s1.metric("Porsi daftar yang ditelepon", f"{kapasitas / len(hasil) * 100:.0f}%")
    s2.metric("Perkiraan deposan didapat", f"{teratas['probabilitas'].sum():.0f}",
              help="Penjumlahan peluang seluruh nasabah yang masuk kapasitas.")
    s3.metric("Dibanding menelepon acak",
              f"+{teratas['probabilitas'].sum() - acak:.0f} deposan",
              delta=f"{C.desimal(teratas['probabilitas'].sum() / max(acak, 1e-9))}x",
              help="Pada jumlah panggilan dan biaya yang persis sama.")
    return kapasitas


def _bukti_urutan(hasil: pd.DataFrame, kapasitas: int):
    """Hanya bisa digambar pada data contoh, karena hasil sebenarnya diketahui."""
    y_true = (hasil["hasil_sebenarnya"] == "yes").astype(int).to_numpy()
    with st.expander("Seberapa bagus urutannya? (bisa dicek karena data contoh "
                     "punya hasil sebenarnya)"):
        x, gain = kurva_gain(y_true, hasil["probabilitas"].to_numpy())
        _, gain_manual = kurva_gain(y_true, hasil["skor_manual"].to_numpy())
        st.plotly_chart(
            charts.kurva_gain_plot({"Model machine learning": (x, gain),
                                    "Skor prioritas manual": (x, gain_manual)},
                                   penanda=[10, 20, 30, 50]),
            use_container_width=True)
        tertangkap = int(y_true[:kapasitas].sum())
        st.caption(
            f"Menelepon {C.ribu(kapasitas)} nasabah teratas menangkap "
            f"**{C.ribu(tertangkap)} dari {C.ribu(y_true.sum())} deposan** "
            f"({tertangkap / max(y_true.sum(), 1) * 100:.0f}%) — dibanding sekitar "
            f"{C.ribu(kapasitas / len(hasil) * y_true.sum())} deposan kalau daftar "
            "ditelepon dengan urutan acak.")


def _tabel_dan_unduh(hasil: pd.DataFrame, teratas: pd.DataFrame, kapasitas: int,
                     ambang: float):
    st.markdown('<div class="langkah"><span>3</span>Lihat dan unduh daftar panggilan</div>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1.1, 1.6, 1.1])
    hanya_kapasitas = f1.toggle(f"Tampilkan {C.ribu(kapasitas)} teratas saja", value=True)
    pilih_prioritas = f2.multiselect(
        "Saring prioritas", [p for p in URUT_PRIORITAS if p in set(hasil["prioritas"])],
        default=[], placeholder="Semua prioritas")
    semua_kolom = f3.toggle("Tampilkan semua kolom", value=False)

    tampil = teratas if hanya_kapasitas else hasil
    if pilih_prioritas:
        tampil = tampil[tampil["prioritas"].isin(pilih_prioritas)]

    tampil = tampil.copy()
    for kol, peta in TERJEMAHAN.items():
        if kol in tampil.columns:
            tampil[kol] = tampil[kol].map(lambda v: peta.get(v, v))

    urut_kolom = [c for c in KOLOM_TAMPIL if c in tampil.columns]
    if semua_kolom:
        urut_kolom += [c for c in tampil.columns
                       if c not in urut_kolom and c not in ("probabilitas", "skor_manual")]

    st.dataframe(
        tampil[urut_kolom].head(1000), hide_index=True, use_container_width=True,
        height=430,
        column_config={
            "peringkat": st.column_config.NumberColumn("No.", width="small"),
            "peluang_persen": st.column_config.ProgressColumn(
                "Peluang buka deposito", format="%.1f%%", min_value=0.0, max_value=100.0,
                help="Peluang nasabah membuka deposito bila ditelepon."),
            "nilai_harapan_eur": st.column_config.NumberColumn(
                "Perkiraan nilai (EUR)", format="%.0f",
                help="Perkiraan hasil satu panggilan setelah dikurangi biayanya."),
            "rekomendasi": st.column_config.TextColumn("Telepon hari ini?"),
            **{k: st.column_config.Column(v) for k, v in LABEL_KOLOM.items()
               if k not in ("peringkat", "peluang_persen", "nilai_harapan_eur",
                            "rekomendasi")},
        })
    if len(tampil) > 1000:
        st.caption(f"Yang tampil di layar 1.000 baris teratas dari "
                   f"{C.ribu(len(tampil))} baris. "
                   "Unduh berkas untuk daftar lengkapnya.")

    kolom_unduh = [c for c in hasil.columns if c != "skor_manual"]
    u1, u2 = st.columns(2)
    u1.download_button(
        f"Unduh {C.ribu(kapasitas)} teratas untuk ditelepon hari ini",
        data=teratas[kolom_unduh].to_csv(index=False).encode("utf-8"),
        file_name=f"call_list_harian_{kapasitas}.csv", mime="text/csv",
        type="primary", use_container_width=True, icon=":material/download:")
    u2.download_button(
        "Unduh seluruh daftar beserta nilainya",
        data=hasil[kolom_unduh].to_csv(index=False).encode("utf-8"),
        file_name="call_list_terskor.csv", mime="text/csv",
        use_container_width=True, icon=":material/download:")
    st.caption("Berkas unduhan memakai kode asli (`cellular`, `may`, ...) agar bisa "
               "langsung dibaca sistem dialer, ditambah kolom peluang dan prioritas.")


def render():
    st.title("Skoring Massal — Daftar Panggilan Harian")
    st.caption("Unggah daftar nasabah, terima antrean panggilan yang sudah terurut "
               "dari yang paling berpeluang.")

    ambang = st.session_state.get("ambang", C.AMBANG_OPERASI)

    df_mentah = _pilih_data()
    if df_mentah is None:
        return

    hasil = _lengkapi(_skor_tabel(df_mentah), ambang)
    st.divider()
    kapasitas = _ringkasan(hasil, ambang)

    if "hasil_sebenarnya" in hasil.columns:
        _bukti_urutan(hasil, kapasitas)

    st.divider()
    _tabel_dan_unduh(hasil, hasil.head(kapasitas), kapasitas, ambang)
