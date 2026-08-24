"""Halaman 3 — skoring massal dan penyusunan call list harian."""
import io

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts, config as C
from ..data import (buat_fitur, muat_data, muat_template, segmen_prioritas,
                    skor_prioritas_manual, validasi_upload)
from ..model import kurva_gain, skor


def _label_segmen(p: float, ambang: float) -> str:
    if p >= max(ambang * 3, 0.5):
        return "A - sangat tinggi"
    if p >= ambang * 2:
        return "B - tinggi"
    if p >= ambang:
        return "C - di atas ambang"
    if p >= ambang / 2:
        return "D - rendah"
    return "E - sangat rendah"


@st.cache_data(show_spinner="Menghitung skor untuk seluruh baris...")
def _skor_tabel(df_mentah: pd.DataFrame, ambang: float) -> pd.DataFrame:
    fitur = buat_fitur(df_mentah)
    prob = skor(fitur)
    hasil = df_mentah.copy()
    hasil.columns = [c.replace(".", "_").strip() for c in hasil.columns]
    hasil["probabilitas"] = prob
    hasil["segmen"] = [_label_segmen(p, ambang) for p in prob]
    hasil["masuk_antrean"] = np.where(prob >= ambang, "Ya", "Tidak")
    hasil["skor_manual"] = skor_prioritas_manual(fitur).to_numpy()
    hasil["nilai_harapan_eur"] = prob * C.NILAI_PER_DEPOSAN - C.BIAYA_PER_PANGGILAN
    hasil = hasil.sort_values("probabilitas", ascending=False).reset_index(drop=True)
    hasil.insert(0, "peringkat", np.arange(1, len(hasil) + 1))
    return hasil


def render():
    st.title("Skoring Massal — Penyusunan Call List")
    st.caption("Unggah daftar nasabah, dapatkan antrean panggilan yang sudah terurut.")

    ambang = st.session_state.get("ambang", C.AMBANG_OPERASI)

    tab_unggah, tab_format = st.tabs(["Unggah & skor", "Format file yang diminta"])

    with tab_format:
        st.markdown("#### Kolom wajib")
        st.markdown(
            "File CSV harus memuat 16 kolom berikut. Kolom tambahan seperti `customer_id` "
            "atau nomor telepon boleh disertakan dan akan ikut dibawa ke hasil."
        )
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
            ("cons_conf_idx", "Angka", "-51 s/d -26", "Consumer confidence index berjalan"),
            ("euribor3m", "Angka", "0,6–5,1", "Suku bunga Euribor 3 bulan berjalan"),
        ], columns=["Kolom", "Tipe", "Nilai yang diterima", "Keterangan"])
        st.dataframe(spek, hide_index=True, use_container_width=True)

        st.download_button("Unduh template CSV (50 baris contoh)",
                           data=muat_template(), file_name="template_call_list.csv",
                           mime="text/csv", icon=":material/download:")
        st.caption("Kolom `duration` sengaja tidak diminta: durasi panggilan baru diketahui "
                   "setelah panggilan selesai, sedangkan model bekerja sebelum panggilan.")

    with tab_unggah:
        sumber = st.radio("Sumber data", ["Unggah file CSV saya", "Pakai contoh data historis"],
                          horizontal=True, label_visibility="collapsed")

        df_mentah = None
        if sumber == "Unggah file CSV saya":
            berkas = st.file_uploader("Pilih file CSV", type=["csv"],
                                      help="Pemisah koma atau titik koma keduanya diterima.")
            if berkas is None:
                st.info("Belum ada file. Unduh template pada tab **Format file yang diminta**, "
                        "atau pilih *Pakai contoh data historis* untuk mencoba aplikasi.",
                        icon=":material/upload_file:")
                return
            isi = berkas.getvalue()
            for pemisah in (",", ";"):
                calon = pd.read_csv(io.BytesIO(isi), sep=pemisah)
                if calon.shape[1] > 1:
                    df_mentah = calon
                    break
            if df_mentah is None:
                st.error("File tidak bisa dibaca sebagai CSV dengan pemisah koma maupun "
                         "titik koma.", icon=":material/error:")
                return

            error, peringatan = validasi_upload(df_mentah)
            if error:
                st.error("File belum bisa diproses:\n\n"
                         + "\n".join(f"- {e}" for e in error), icon=":material/error:")
                return
            if peringatan:
                with st.expander(f"{len(peringatan)} peringatan pada file (tidak menggagalkan)"):
                    for w in peringatan:
                        st.markdown(f"- {w}")
            st.success(f"File terbaca: **{len(df_mentah):,} baris**, "
                       f"{df_mentah.shape[1]} kolom.", icon=":material/check_circle:")
        else:
            n = st.slider("Jumlah nasabah contoh", 500, 8000, 3000, 500)
            df_pop = muat_data()
            df_mentah = (df_pop.sample(n, random_state=42)
                         .reset_index(drop=True)[C.KOLOM_WAJIB_UPLOAD + ["y"]])
            df_mentah.insert(0, "customer_id",
                             [f"CUST-{i:05d}" for i in range(1, len(df_mentah) + 1)])
            st.caption(f"Contoh {n:,} nasabah diambil acak dari data historis. Kolom `y` "
                       "(hasil sebenarnya) ikut dibawa hanya untuk memvalidasi kualitas urutan; "
                       "pada penggunaan nyata kolom ini tentu belum ada.")

        hasil = _skor_tabel(df_mentah, ambang)

        st.divider()
        st.markdown("### Ringkasan antrean")
        k1, k2, k3, k4 = st.columns(4)
        n_antre = int((hasil["probabilitas"] >= ambang).sum())
        k1.metric("Nasabah diskor", f"{len(hasil):,}")
        k2.metric(f"Di atas ambang {ambang:.2f}", f"{n_antre:,}",
                  delta=f"{n_antre / len(hasil) * 100:.1f}% dari daftar")
        k3.metric("Peluang tertinggi", f"{hasil['probabilitas'].max() * 100:.1f}%")
        k4.metric("Rata-rata peluang", f"{hasil['probabilitas'].mean() * 100:.1f}%")

        st.markdown("### Simulasi kapasitas panggilan")
        kapasitas = st.slider(
            "Berapa panggilan yang sanggup dilakukan hari ini?",
            min_value=min(50, len(hasil)), max_value=len(hasil),
            value=min(max(len(hasil) // 5, 50), len(hasil)),
            step=max(len(hasil) // 100, 1),
            help="Aplikasi menampilkan siapa saja yang masuk ke dalam kapasitas tersebut.")

        teratas = hasil.head(kapasitas)
        s1, s2, s3 = st.columns(3)
        s1.metric("Porsi daftar ditelepon", f"{kapasitas / len(hasil) * 100:.1f}%")
        s2.metric("Perkiraan deposan didapat", f"{teratas['probabilitas'].sum():.0f}",
                  help="Penjumlahan probabilitas seluruh nasabah dalam kapasitas.")
        acak = kapasitas * hasil["probabilitas"].mean()
        s3.metric("Dibanding urutan acak", f"+{teratas['probabilitas'].sum() - acak:.0f} deposan",
                  delta=f"{teratas['probabilitas'].sum() / max(acak, 1e-9):.2f}x",
                  help="Pada jumlah panggilan dan biaya yang persis sama.")

        if "y" in hasil.columns:
            y_true = (hasil["y"] == "yes").astype(int).to_numpy()
            x, gain = kurva_gain(y_true, hasil["probabilitas"].to_numpy())
            _, gain_manual = kurva_gain(y_true, hasil["skor_manual"].to_numpy())
            st.plotly_chart(
                charts.kurva_gain_plot(
                    {"Model machine learning": (x, gain),
                     "Skor prioritas manual": (x, gain_manual)},
                    penanda=[10, 20, 30, 50]),
                use_container_width=True)
            tertangkap = int(y_true[np.argsort(hasil['probabilitas'].to_numpy())[::-1]][:kapasitas].sum())
            st.caption(
                f"Pada contoh ini, menelepon {kapasitas:,} nasabah teratas menangkap "
                f"**{tertangkap:,} dari {int(y_true.sum()):,} deposan** "
                f"({tertangkap / max(y_true.sum(), 1) * 100:.1f}%). Kurva ini hanya bisa "
                "digambar karena hasil sebenarnya diketahui pada data contoh.")

        st.markdown("### Daftar panggilan terurut")
        f1, f2 = st.columns([1, 2])
        hanya_antre = f1.toggle("Tampilkan yang di atas ambang saja", value=False)
        segmen_pilih = f2.multiselect("Saring segmen",
                                      sorted(hasil["segmen"].unique()), default=[])

        tampil = hasil.copy()
        if hanya_antre:
            tampil = tampil[tampil["probabilitas"] >= ambang]
        if segmen_pilih:
            tampil = tampil[tampil["segmen"].isin(segmen_pilih)]

        kolom_depan = [c for c in ["peringkat", "customer_id", "probabilitas", "segmen",
                                   "masuk_antrean", "nilai_harapan_eur", "age", "job",
                                   "contact", "month", "poutcome", "euribor3m"]
                       if c in tampil.columns]
        sisa = [c for c in tampil.columns if c not in kolom_depan]

        st.dataframe(
            tampil[kolom_depan + sisa].head(1000),
            hide_index=True, use_container_width=True, height=420,
            column_config={
                "peringkat": st.column_config.NumberColumn("Urutan", width="small"),
                "probabilitas": st.column_config.ProgressColumn(
                    "Peluang konversi", format="%.3f", min_value=0.0, max_value=1.0),
                "nilai_harapan_eur": st.column_config.NumberColumn(
                    "Nilai harapan (EUR)", format="%.2f"),
                "segmen": st.column_config.TextColumn("Segmen"),
                "masuk_antrean": st.column_config.TextColumn("Ditelepon?", width="small"),
            })
        if len(tampil) > 1000:
            st.caption(f"Menampilkan 1.000 baris teratas dari {len(tampil):,}. "
                       "Unduh berkas untuk daftar lengkap.")

        u1, u2 = st.columns(2)
        u1.download_button(
            "Unduh seluruh daftar terskor",
            data=hasil.to_csv(index=False).encode("utf-8"),
            file_name="call_list_terskor.csv", mime="text/csv",
            use_container_width=True, icon=":material/download:")
        u2.download_button(
            f"Unduh {kapasitas:,} teratas untuk dialer hari ini",
            data=teratas.to_csv(index=False).encode("utf-8"),
            file_name=f"call_list_harian_{kapasitas}.csv", mime="text/csv",
            type="primary", use_container_width=True, icon=":material/download:")
