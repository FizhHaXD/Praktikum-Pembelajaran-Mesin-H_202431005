# Praktikum Pembelajaran Mesin (Kelas H)

Repositori ini berisi kode dan dokumentasi praktikum mata kuliah **Pembelajaran Mesin** (Kelas H), program studi S1 Teknik Informatika, Institut Teknologi PLN.

Praktikum yang dibahas mencakup dua materi utama:
1. **Praktek 1 — Exploratory Data Analysis (EDA)**
2. **Praktikum Regresi Linear (Sederhana & Berganda)**

---

## Identitas Mahasiswa

| Informasi | Keterangan |
|---|---|
| **Nama** | Muhammad Hafizh Wijdan |
| **NIM** | 202431005 |
| **Mata Kuliah** | Pembelajaran Mesin |
| **Kelas** | H |
| **Program Studi** | S1 Teknik Informatika |
| **Institusi** | Institut Teknologi PLN |

---

## Struktur File

```text
├── Praktek 1 - EDA.ipynb                 # Notebook Praktek 1: Exploratory Data Analysis (EDA)
├── Regresi LINEAR.ipynb                  # Notebook Regresi Linear Sederhana & Berganda
├── healthcare-dataset-stroke-data.csv    # Dataset kasus medis stroke (untuk EDA)
├── botak.csv                             # Dataset kasus probabilitas kebotakan (untuk Regresi Berganda)
├── LAPORAN_PRAKTIKUM_MACHINE_LEARNING.md # Laporan lengkap hasil praktikum EDA & Regresi
├── .gitignore                            # Berkas pengecualian Git
└── README.md                             # Dokumentasi repositori
```

---

## 1. Praktek 1 — Exploratory Data Analysis (EDA)

Praktikum ini berfokus pada tahapan awal pengenalan dan pemahaman dataset sebelum proses pemodelan machine learning. Dataset yang digunakan adalah data klinis stroke (`healthcare-dataset-stroke-data.csv`) serta eksplorasi dasar dataset Iris.

### Tahapan yang Dikerjakan:
- **Inspeksi Awal:** Memeriksa ukuran matriks (`df.shape`), tipe data setiap kolom (`df.info()`), dan ringkasan nilai statistik deskriptif (`df.describe()`).
- **Analisis Distribusi Kelas:** Memeriksa perbandingan data target `stroke` menggunakan countplot untuk mengidentifikasi kondisi ketimpangan kelas (*class imbalance*).
- **Penanganan Missing Values:** Mengidentifikasi data kosong pada kolom `bmi` dan melakukan imputasi menggunakan nilai median kolom.
- **Deteksi Outlier:** Visualisasi sebaran kolom `age` menggunakan boxplot dan perhitungan rentang Interquartile Range (IQR).
- **Analisis Multivariat & Bivariat:** Membuat correlation heatmap untuk melihat relasi linear antar fitur numerik, serta analisis hubungan umur dan kadar glukosa terhadap kejadian stroke.

---

## 2. Praktikum Regresi Linear

Praktikum ini membahas penerapan algoritma pembelajaran terawasi (*Supervised Learning*) menggunakan model regresi linear. Terbagi menjadi dua implementasi:

### A. Regresi Linear Sederhana (Bivariat)
- Menggunakan data relasi antara tinggi badan ($X$) terhadap berat badan ($y$).
- Menghitung korelasi awal dengan `df.corr()`.
- Melatih model `LinearRegression` dari scikit-learn menggunakan metode *Ordinary Least Squares* (OLS).
- Melakukan pengujian prediksi pada data tinggi badan baru.
- Menarik garis linear terbaik (*best-fit line*) antara data aktual dan nilai prediksi.
- Evaluasi kinerja model menggunakan skor $R^2$.

### B. Regresi Linear Berganda (Multivariat)
- Menggunakan dataset `botak.csv` untuk memprediksi probabilitas kebotakan (`botak_prob`).
- Pembersihan data awal dengan menghapus baris yang mengandung nilai kosong (`dropna`).
- Pemisahan fitur independen ($X$: `umur`, `gaji`, `tinggi`, `berat`) dan variabel target ($y$: `botak_prob`).
- Pembagian data latih dan data uji menggunakan *Train-Test Split* dengan proporsi 80:20 (`random_state=42`).
- Pelatihan model `LinearRegression` multivariat.
- Evaluasi performa model menggunakan metrik *Mean Squared Error* (MSE) dan koefisien determinasi ($R^2$ Score).
- Visualisasi evaluasi melalui scatter plot antara nilai aktual target uji ($y_{\text{test}}$) dengan hasil prediksi ($y_{\text{pred}}$).

---

## Kebutuhan Library & Lingkungan Kerja

Praktikum dijalankan menggunakan Python 3.x dan Jupyter Notebook. Library yang digunakan meliputi:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Instalasi library dapat dilakukan melalui pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```
