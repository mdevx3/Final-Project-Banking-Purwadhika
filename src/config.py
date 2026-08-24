"""Konstanta aplikasi — seluruhnya diturunkan dari notebook Final Project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "best_model.pkl"
DATA_PATH = ROOT / "data" / "bank_marketing_clean.csv.gz"
TEMPLATE_PATH = ROOT / "data" / "template_call_list.csv"

# --- Kontrak fitur model (Section F.2 notebook) -----------------------------
FITUR_NUMERIK = ["campaign", "previous", "pdays_clean", "euribor3m", "cons_conf_idx",
                 "was_contacted_before", "has_previous_contact"]
FITUR_ORDINAL = ["education"]
FITUR_NOMINAL = ["job", "marital", "default", "housing", "loan", "contact",
                 "month", "day_of_week", "poutcome", "age_group"]
FITUR_MODEL = FITUR_NUMERIK + FITUR_ORDINAL + FITUR_NOMINAL

# Kolom mentah yang wajib ada pada file yang diunggah (fitur turunan dihitung app)
KOLOM_WAJIB_UPLOAD = ["age", "job", "marital", "education", "default", "housing", "loan",
                      "contact", "month", "day_of_week", "campaign", "pdays", "previous",
                      "poutcome", "cons_conf_idx", "euribor3m"]

# --- Titik kerja model (Section F.7 notebook) ------------------------------
AMBANG_OPERASI = 0.13          # dicari lewat CV pada data latih, bukan dipatok 0,5

# --- Asumsi ekonomi (Section A Business Understanding) ---------------------
BIAYA_PER_PANGGILAN = 1.5      # EUR — 0,05 EUR/menit x 30 menit
NILAI_PER_DEPOSAN = 5000.0     # EUR — penempatan minimal per deposito

# --- Pilihan pdays pada formulir -------------------------------------------
# Sentinel 999 ditampilkan sebagai teks, bukan angka, supaya pengguna tidak perlu
# tahu kodenya. Rentang 0-27 mengikuti nilai yang benar-benar ada pada data latih.
PDAYS_BELUM_PERNAH = "Belum pernah"
OPSI_PDAYS = [PDAYS_BELUM_PERNAH] + [str(i) for i in range(0, 28)]

# --- Pengelompokan usia (Section D.c) --------------------------------------
BIN_USIA = [0, 25, 35, 45, 55, 65, 100]
LABEL_USIA = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]

# --- Kategori yang dikenali encoder (diambil dari model yang sudah di-fit) --
URUTAN_EDUKASI = ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school",
                  "professional.course", "university.degree", "unknown"]
URUT_BULAN = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
URUT_HARI = ["mon", "tue", "wed", "thu", "fri"]

BULAN_EMAS = ["mar", "sep", "oct", "dec"]

# --- Label ramah-pengguna ---------------------------------------------------
LABEL_JOB = {
    "admin.": "Administrasi", "blue-collar": "Buruh / pekerja kasar",
    "entrepreneur": "Wiraswasta", "housemaid": "Asisten rumah tangga",
    "management": "Manajemen", "retired": "Pensiunan",
    "self-employed": "Pekerja mandiri", "services": "Jasa / layanan",
    "student": "Pelajar / mahasiswa", "technician": "Teknisi",
    "unemployed": "Tidak bekerja", "unknown": "Tidak diketahui",
}
LABEL_MARITAL = {"divorced": "Cerai", "married": "Menikah",
                 "single": "Lajang", "unknown": "Tidak diketahui"}
LABEL_EDUKASI = {
    "illiterate": "Tidak bersekolah", "basic.4y": "SD (4 tahun)",
    "basic.6y": "SD (6 tahun)", "basic.9y": "SMP (9 tahun)",
    "high.school": "SMA", "professional.course": "Kursus profesional",
    "university.degree": "Sarjana", "unknown": "Tidak diketahui",
}
LABEL_YNU = {"no": "Tidak", "yes": "Ya", "unknown": "Tidak diketahui"}
LABEL_CONTACT = {"cellular": "Telepon seluler", "telephone": "Telepon rumah"}
LABEL_POUTCOME = {"success": "Berhasil (pernah ambil deposito)",
                  "failure": "Gagal (pernah ditawari, menolak)",
                  "nonexistent": "Belum pernah dihubungi"}
LABEL_BULAN = {"jan": "Januari", "feb": "Februari", "mar": "Maret", "apr": "April",
               "may": "Mei", "jun": "Juni", "jul": "Juli", "aug": "Agustus",
               "sep": "September", "oct": "Oktober", "nov": "November", "dec": "Desember"}
LABEL_HARI = {"mon": "Senin", "tue": "Selasa", "wed": "Rabu",
              "thu": "Kamis", "fri": "Jumat"}

# --- Warna ------------------------------------------------------------------
HIJAU = "#2E8B57"
MERAH = "#B22222"
BIRU = "#4682B4"
ABU = "#8C8C8C"


# --- Format angka ------------------------------------------------------------
def ribu(n: float) -> str:
    """Pemisah ribuan gaya Indonesia: 3000 -> '3.000'.

    Dipakai pada angka yang dibaca pengguna. Tanpa ini "3,000" terbaca sebagai
    tiga koma nol bagi pembaca Indonesia.
    """
    return f"{n:,.0f}".replace(",", ".")


def desimal(n: float, angka: int = 1) -> str:
    """Angka desimal gaya Indonesia: 13.0 -> '13,0'."""
    return f"{n:.{angka}f}".replace(".", ",")
