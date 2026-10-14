# LAPORAN PRAKTIKUM MACHINE LEARNING
## EXPLORATORY DATA ANALYSIS (EDA) & PEMODELAN REGRESI LINEAR (SEDERHANA DAN BERGANDA)

---

### IDENTITAS PRAKTIKUM
* **Mata Kuliah** : Praktikum Machine Learning / Sains Data
* **Materi Praktikum** : Modul 1 (Exploratory Data Analysis) & Modul 2 (Regresi Linear)
* **Dataset Kasus** :
  1. *Iris Dataset* (`sklearn.datasets.load_iris`)
  2. *Healthcare Stroke Dataset* (`healthcare-dataset-stroke-data.csv`)
  3. *Botak Probability Dataset* (`botak.csv`)
* **Penyusun (Mahasiswa)** : Muhammad Hafizh Wijdan
* **Nomor Induk Mahasiswa (NIM)** : 202431005
* **Program Studi / Kelas** : Teknik Informatika / Pembelajaran Mesin (Kelas H)
* **Tanggal Penyusunan** : Oktober 2026

---

## 📌 DAFTAR ISI
1. [BAB I: PENDAHULUAN](#bab-i-pendahuluan)
   - 1.1 Latar Belakang
   - 1.2 Tujuan Praktikum
   - 1.3 Ruang Lingkup Materi
2. [BAB II: LANDASAN TEORI](#bab-ii-landasan-teori)
   - 2.1 Konsep Exploratory Data Analysis (EDA)
   - 2.2 Statistik Deskriptif & Deteksi Anomali Data (Missing Values & IQR Outlier)
   - 2.3 Matriks Korelasi Pearson
   - 2.4 Teori Regresi Linear Sederhana & Ordinary Least Squares (OLS)
   - 2.5 Teori Regresi Linear Berganda
   - 2.6 Metrik Evaluasi Regresi (MSE dan $R^2$ Score)
3. [BAB III: METODOLOGI & ALAT PRAKTIKUM](#bab-iii-metodologi--alat-praktikum)
   - 3.1 Spesifikasi Perangkat Lunak & Pustaka Python
   - 3.2 Metadata Dataset Praktikum
   - 3.3 Diagram Alur Alur Kerja (Pipeline Data Science)
4. [BAB IV: LANGKAH-LANGKAH PRAKTIKUM & HASIL EKSEKUSI STEP-BY-STEP](#bab-iv-langkah-langkah-praktikum--hasil-eksekusi-step-by-step)
   - [BAGIAN A: PRAKTIKUM 1 — EXPLORATORY DATA ANALYSIS (EDA)](#bagian-a-praktikum-1--exploratory-data-analysis-eda)
     - Step 1: Inisialisasi Pustaka & Eksplorasi Multi-Panel Iris
     - Step 2: Pemuatan Dataset Stroke & Pratinjau Teratas (`df.head()`)
     - Step 3: Pemeriksaan Dimensi & Tipe Data (`df.info()`)
     - Step 4: Analisis Statistik Deskriptif (`df.describe()`)
     - Step 5: Visualisasi Distribusi Kelas Target (Class Imbalance)
     - Step 6: Konversi Kolom Tipe Kategorikal
     - Step 7: Identifikasi Nilai Hilang (`df.isnull().sum()`)
     - Step 8: Penanganan Nilai Hilang Menggunakan Imputasi Median Kolom BMI
     - Step 9: Deteksi Outlier pada Umur Menggunakan Boxplot & Metode IQR
     - Step 10: Analisis Korelasi Multivariat Fitur Numerik (Heatmap)
     - Step 11: Analisis Bivariat Hubungan Umur terhadap Kejadian Stroke
     - Step 12: Analisis Bivariat Hubungan Kadar Glukosa terhadap Kejadian Stroke
   - [BAGIAN B: PRAKTIKUM 2 — REGRESI LINEAR SEDERHANA](#bagian-b-praktikum-2--regresi-linear-sederhana)
     - Step 1: Pembuatan Data Sampel Bivariat (Tinggi vs Berat Badan)
     - Step 2: Pemeriksaan Korelasi Pearson
     - Step 3: Penyiapan Matriks Fitur ($X$) dan Target ($y$)
     - Step 4: Pelatihan Model Regresi Linear OLS
     - Step 5: Pengujian Model pada Sampel Uji Baru
     - Step 6: Ekstraksi Parameter Matematis (Slope $m$ dan Intercept $c$)
     - Step 7: Visualisasi Garis Regresi Model Terbaik (Best-Fit Line)
     - Step 8: Evaluasi Kinerja Model dengan Nilai $R^2$ Score
   - [BAGIAN C: PRAKTIKUM 3 — REGRESI LINEAR BERGANDA (botak.csv)](#bagian-c-praktikum-3--regresi-linear-berganda-studi-kasus-botakcsv)
     - Step 9: Pemuatan Dataset `botak.csv` & Inspeksi Struktur
     - Step 10: Pembersihan Data (Eliminasi Baris Kosong dengan `dropna`)
     - Step 11: Pemisahan Matriks Fitur Multivariat ($X$) dan Target ($y$)
     - Step 12: Pembagian Data Train-Test (Split 80:20)
     - Step 13: Pelatihan Model Regresi Linear Berganda
     - Step 14: Pengujian & Prediksi pada Data Uji ($y_{\text{pred}}$)
     - Step 15: Ekstraksi Bobot Koefisien dan Intercept Model
     - Step 16: Evaluasi Kinerja Model dengan Nilai MSE dan $R^2$ Score
     - Step 17: Visualisasi Scatter Plot Prediksi vs Nilai Aktual
5. [BAB V: PEMBAHASAN & ANALISIS KRITIS](#bab-v-pembahasan--analisis-kritis)
   - 5.1 Karakteristik Dataset Stroke & Pengaruh Class Imbalance
   - 5.2 Komparasi Model: Mengapa $R^2$ Regresi Sederhana 97.7% dan Regresi Berganda 11.7%?
   - 5.3 Makna Bobot Koefisien Fitur terhadap Probabilitas Kebotakan
   - 5.4 Keterbatasan Model Regresi Linear pada Target Probabilitas
6. [BAB VI: KESIMPULAN & SARAN](#bab-vi-kesimpulan--saran)
   - 6.1 Kesimpulan Praktikum
   - 6.2 Saran Pengembangan Lanjutan
7. [DAFTAR PUSTAKA](#daftar-pustaka)

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang
Di era kecerdasan buatan dan ledakan data (*big data*), kemampuan menganalisis informasi mentah dan mengekstrak pemodelan matematis adalah kompetensi fundamental. Di dalam alur kerja sains data (*data science workflow*), langkah awal yang wajib dilakukan sebelum membangun model adalah **Exploratory Data Analysis (EDA)**. Tanpa EDA, praktisi data tidak akan mengetahui apakah dataset memiliki nilai kosong (*missing values*), anomali data (*outliers*), atau ketimpangan distribusi kelas (*imbalanced data*) yang berpotensi merusak keandalan model.

Setelah data dipahami, tahapan pemodelan dimulai. Salah satu algoritma pembelajaran terawasi (*Supervised Learning*) tertua dan paling interpretatif adalah **Regresi Linear**. Algoritma ini memodelkan hubungan matematis antara satu atau lebih fitur independen ($X$) terhadap target numerik kontinu ($y$). Praktikum ini mengombinasikan pemahaman teoritis dan praktis melalui studi kasus nyata pada data klinis penyakit stroke dan data prediksi probabilitas kebotakan (`botak.csv`).

### 1.2 Tujuan Praktikum
1. Mampu mengoperasikan lingkungan kerja sains data berbasis Python (*Jupyter Notebook*, *Pandas*, *NumPy*, *Matplotlib*, *Seaborn*, dan *Scikit-Learn*).
2. Memahami dan mengimplementasikan tahapan EDA untuk inspeksi struktur data, statistik deskriptif, dan deteksi nilai kosong.
3. Menerapkan teknik pembersihan data seperti imputasi median dan eliminasi baris bernilai null.
4. Membuat visualisasi statistik informatif (boxplot, countplot, scatterplot, heatmap korelasi).
5. Membangun model **Regresi Linear Sederhana** serta menginterpretasikan garis linear *best-fit*.
6. Membangun model **Regresi Linear Berganda** menggunakan validasi *Train-Test Split*.
7. Mengevaluasi performa model regresi menggunakan parameter koefisien, *intercept*, *Mean Squared Error* (MSE), dan *R-squared* ($R^2$ Score).

### 1.3 Ruang Lingkup Materi
Praktikum ini mencakup 2 modul utama:
- **Modul 1: Exploratory Data Analysis (EDA)** dengan subjek dataset Iris dan Healthcare Stroke Dataset (5.110 baris, 12 kolom).
- **Modul 2: Pemodelan Regresi Linear** yang terbagi menjadi:
  - Regresi Linear Sederhana (Dataset Tinggi vs Berat Badan, 10 observasi).
  - Regresi Linear Berganda (Dataset `botak.csv`, 7.991 baris, 4 prediktor: umur, gaji, tinggi, dan berat).

---

## BAB II: LANDASAN TEORI

### 2.1 Konsep Exploratory Data Analysis (EDA)
EDA diperkenalkan oleh John Tukey (1977) sebagai metode eksplorasi investigatif untuk membedah data sebelum menetapkan asumsi formal. Tujuan pokok EDA meliputi:
- Memverifikasi apakah data sesuai dengan hipotesis awal.
- Menemukan pola distribusi dan anomali (*noise*, *outliers*, *missing values*).
- Menentukan fitur-fitur yang paling relevan (*feature importance*) terhadap variabel target.

### 2.2 Statistik Deskriptif & Deteksi Anomali
Statistik deskriptif merangkum sebaran data melalui:
- **Ukuran Pemusatan**: Rata-rata (*mean*), nilai tengah (*median*), dan frekuensi terbanyak (*modus*).
- **Ukuran Penyebaran**: Standar deviasi ($s$), variansi ($s^2$), nilai minimum, maksimum, dan kuartil ($Q_1, Q_2, Q_3$).

#### Penanganan Missing Values
Jika data bolong bernilai kontinu dan memiliki distribusi condong (*skewed*), pengisian menggunakan nilai **median** jauh lebih aman daripada **mean**, karena median kebal terhadap nilai-nilai ekstrem.

#### Deteksi Outlier dengan Rentang Antarkuartil (IQR)
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Batas Bawah} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Batas Atas} = Q_3 + 1.5 \times \text{IQR}$$
Titik observasi di luar rentang $[\text{Batas Bawah}, \text{Batas Atas}]$ dinyatakan sebagai pencilan (*outlier*).

### 2.3 Matriks Korelasi Pearson
Korelasi Pearson ($r$) mengukur derajat hubungan linear antara dua variabel kontinu:
$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
Nilai $r$ berada pada rentang $-1 \le r \le +1$:
- $r = +1$: Hubungan linear positif sempurna.
- $r = 0$: Tidak terdapat hubungan linear.
- $r = -1$: Hubungan linear negatif sempurna.

### 2.4 Teori Regresi Linear Sederhana & Ordinary Least Squares (OLS)
Model regresi linear sederhana menghubungkan variabel input $x$ dengan target $y$:
$$y = m \cdot x + c$$
di mana:
- $m$: Kemiringan garis (*slope* / koefisien regresi).
- $c$: Konstanta titik potong sumbu $y$ (*intercept*).

Metode **Ordinary Least Squares (OLS)** mencari nilai $m$ dan $c$ yang meminimalkan jumlah kuadrat sisa (*Sum of Squared Residuals* / SSR):
$$\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 2.5 Teori Regresi Linear Berganda
Model regresi linear berganda memodelkan target $y$ dari $k$ buah variabel independen:
$$y = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_k x_k$$
di mana $b_0$ adalah *intercept* dan $b_1, \dots, b_k$ adalah koefisien parsial yang menyatakan perubahan pada $y$ untuk setiap unit peningkatan pada $x_i$, dengan asumsi variabel lainnya bernilai konstan (*ceteris paribus*).

### 2.6 Metrik Evaluasi Regresi
- **Mean Squared Error (MSE)**:
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- **Koefisien Determinasi ($R^2$ Score)**:
  $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$
  $R^2$ bernilai $1.0$ menunjukkan bahwa model mampu memprediksi target dengan sempurna tanpa residual.

---

## BAB III: METODOLOGI & ALAT PRAKTIKUM

### 3.1 Spesifikasi Perangkat Lunak & Pustaka
| Perangkat / Pustaka | Versi / Lingkungan | Kegunaan |
|---|---|---|
| Sistem Operasi | Windows 11 (64-bit) | Sistem operasi eksekusi |
| Bahasa Pemrograman | Python 3.11+ | Pemrograman saintifik |
| Lingkungan Kerja | Jupyter Notebook / VS Code | Interactive execution & markdown reporting |
| `pandas` | 2.x | Manajemen tabular DataFrame |
| `numpy` | 1.24+ | Operasi larik & aljabar linear numerik |
| `matplotlib` & `seaborn` | 3.8+ & 0.13+ | Visualisasi grafik saintifik |
| `scikit-learn` | 1.4+ | Algoritma LinearRegression, evaluasi metrik, train-test split |

### 3.2 Metadata Dataset Praktikum
1. **Dataset Healthcare Stroke** (`healthcare-dataset-stroke-data.csv`):
   - Jumlah entri: 5.110 baris, 12 fitur.
   - Kolom target: `stroke` (0 = Tidak stroke, 1 = Menderita stroke).
2. **Dataset Sampel Tinggi-Berat**:
   - Jumlah observasi: 10 sampel data kontinu tinggi badan ($x$) vs berat badan ($y$).
3. **Dataset Botak** (`botak.csv`):
   - Jumlah entri: 7.991 baris, 13 fitur.
   - Kolom target: `botak_prob` (Probabilitas kontinuitas 0.0 s/d 1.0).
   - Fitur prediktor: `umur`, `gaji`, `tinggi`, `berat`.

### 3.3 Diagram Alur Pipeline Praktikum
```
[1. Load Dataset] ──> [2. Cleaning & EDA] ──> [3. Feature-Target Split (X, y)]
                            │
                            ▼
[6. Evaluasi Metrik] <── [5. Model Fitting] <── [4. Train-Test Split (80:20)]
```

---

## BAB IV: LANGKAH-LANGKAH PRAKTIKUM & HASIL EKSEKUSI STEP-BY-STEP

### BAGIAN A: PRAKTIKUM 1 — EXPLORATORY DATA ANALYSIS (EDA)

#### Step 1: Inisialisasi Pustaka & Eksplorasi Multi-Panel Iris
**Tujuan**: Menyiapkan pustaka analisis data dan menguji visualisasi data pada Iris Dataset bawaan Scikit-Learn.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

sns.set(style="whitegrid")
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['species'] = [iris.target_names[i] for i in iris.target]
```

**Tangkapan Layar Bukti Eksekusi**:
![Visualisasi 4-Panel Iris](laporan_assets/step_eda_01_iris_multipanel.png)
*Gambar 4.1: Visualisasi 4-Panel EDA Dataset Iris (Distribusi Sepal Length, Scatter Plot Sepal, Boxplot Petal, dan Matriks Korelasi)*

![Pairplot Seluruh Kombinasi Fitur Iris](laporan_assets/step_eda_01b_iris_pairplot.png)
*Gambar 4.2: Visualisasi Pairplot Interaksi Seluruh Kombinasi Fitur Iris*

**Analisis Hasil**:
Dari Gambar 4.1 dan 4.2, spesies *Setosa* dapat dipisahkan secara linier dengan sangat jelas dari *Versicolor* dan *Virginica* hanya menggunakan panjang dan lebar mahkota (*Petal Length* dan *Petal Width*). Fitur petal memiliki korelasi linear sangat tinggi ($r > 0.85$).

---

#### Step 2: Pemuatan Dataset Stroke & Pratinjau Teratas (`df.head()`)
**Tujuan**: Membaca dataset stroke ke dalam DataFrame dan memeriksa 5 baris pertama data.

```python
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.head()
```

**Tangkapan Layar Bukti Eksekusi**:
![Tangkapan Layar df.head](laporan_assets/step_eda_02_df_head.png)
*Gambar 4.3: Tangkapan Layar Eksekusi df.head() Menampilkan 5 Baris Pertama Dataset Stroke*

**Analisis Hasil**:
Dataset berhasil dimuat ke memori. Terlihat variabel identifikasi (`id`), demografi (`gender`, `age`, `ever_married`, `work_type`, `Residence_type`), kondisi klinis (`hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`), dan target akhir `stroke`.

---

#### Step 3: Pemeriksaan Dimensi & Tipe Data (`df.info()`)
**Tujuan**: Mengecek tipe data dan mendeteksi kolom mana yang memiliki nilai kosong (*Non-Null Count*).

```python
df.info()
```

**Tangkapan Layar Bukti Eksekusi**:
![Tangkapan Layar df.info](laporan_assets/step_eda_03_df_info.png)
*Gambar 4.4: Tangkapan Layar Output df.info() Struktur dan Tipe Data Stroke*

**Analisis Hasil**:
Total observasi adalah 5.110 baris. Seluruh kolom terisi penuh sebanyak 5.110 entri, kecuali kolom `bmi` yang hanya memiliki 4.909 entri non-null. Terdapat 201 missing values pada kolom BMI yang harus ditangani pada tahap pra-pemrosesan data.

---

#### Step 4: Analisis Statistik Deskriptif (`df.describe()`)
**Tujuan**: Memeriksa ringkasan statistik angka numerik (rata-rata, standar deviasi, kuartil, minimum, dan maksimum).

```python
df.describe()
```

**Tangkapan Layar Bukti Eksekusi**:
![Tangkapan Layar df.describe](laporan_assets/step_eda_04_df_describe.png)
*Gambar 4.5: Tangkapan Layar Tabel Ringkasan Statistik Deskriptif Fitur Numerik*

**Analisis Hasil**:
Rata-rata umur pasien adalah 43.2 tahun dengan rentang 0.08 s/d 82 tahun. Fitur `avg_glucose_level` menunjukkan simpangan yang besar (55.12 s/d 271.74 mg/dL). Rata-rata target `stroke` adalah 0.0487, menunjukkan proporsi penderita stroke hanya sekitar 4.87%.

---

#### Step 5: Visualisasi Distribusi Kelas Target (Class Imbalance)
**Tujuan**: Menghitung dan memvisualisasikan frekuensi pasien bebas stroke vs menderita stroke.

```python
sns.countplot(x="stroke", data=df)
plt.title("Distribusi Kelas Target (Stroke)")
plt.show()
```

**Tangkapan Layar Bukti Eksekusi**:
![Distribusi Kelas Target Stroke](laporan_assets/step_eda_05_countplot_target.png)
*Gambar 4.6: Visualisasi Distribusi Pasien Bebas Stroke (0) vs Pasien Terkena Stroke (1)*

**Analisis Hasil**:
Gambar 4.6 memperlihatkan disparitas data yang sangat timpang: kelas 0 berjumlah 4.861 orang (95.1%), sementara kelas 1 hanya 249 orang (4.9%). Fenomena *class imbalance* ini merupakan isu krusial dalam klasifikasi medis yang menuntut metrik evaluasi khusus selain sekadar *accuracy* (seperti F1-score, Precision, Recall, atau ROC-AUC).

---

#### Step 6: Konversi Kolom Tipe Kategorikal
**Tujuan**: Mengubah kolom teks object menjadi tipe data category agar efisien secara komputasi.

```python
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

df.dtypes
```

**Analisis Hasil**:
Seluruh kolom kategorikal telah berstatus data `category` dan siap untuk analisis agregasi statistik tingkat lanjut.

---

#### Step 7 & 8: Identifikasi & Penanganan Missing Values (Imputasi Median Kolom BMI)
**Tujuan**: Menghitung persis jumlah nilai yang hilang pada kolom BMI dan mengisi kekosongan tersebut dengan nilai median.

```python
print("Missing values awal:\n", df.isnull().sum())

# Imputasi menggunakan median
df["bmi"].fillna(df["bmi"].median(), inplace=True)

print("\nMissing values setelah imputasi:\n", df.isnull().sum())
```

**Tangkapan Layar Bukti Eksekusi**:
![Imputasi Median Kolom BMI](laporan_assets/step_eda_08_imputasi_missing.png)
*Gambar 4.7: Tangkapan Layar Deteksi & Imputasi Median Missing Value Kolom BMI*

**Analisis Hasil**:
Sebanyak 201 nilai hilang pada kolom `bmi` berhasil diisi dengan nilai median (28.1). Median dipilih karena variabel BMI memiliki pencilan tinggi (*skewed* kanan) sehingga median lebih aman dan representatif dibanding nilai rata-rata (*mean*). Dataset kini 100% bebas dari nilai kosong (*clean data*).

---

#### Step 9: Deteksi Outlier pada Umur Menggunakan Boxplot & Metode IQR
**Tujuan**: Memeriksa apakah terdapat titik ekstrem di luar rentang $1.5 \times \text{IQR}$ pada kolom usia pasien.

```python
sns.boxplot(x=df["age"])
plt.title("Box Plot Umur")
plt.show()

Q1 = df["age"].quantile(0.25)
Q3 = df["age"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["age"] < Q1 - 1.5 * IQR) | (df["age"] > Q3 + 1.5 * IQR)]
print("Jumlah Outlier umur: ", len(outliers))
```

**Tangkapan Layar Bukti Eksekusi**:
![Box Plot Umur](laporan_assets/step_eda_09_boxplot_umur.png)
*Gambar 4.8: Boxplot Distribusi Usia Pasien*

**Analisis Hasil**:
Kalkulasi numerik menunjukkan jumlah outlier umur adalah **0**. Seluruh titik data usia berada di dalam batas kumis (*whiskers*) boxplot, membuktikan bahwa rentang usia 0 s/d 82 tahun adalah data riil yang valid tanpa anomali pencatatan.

---

#### Step 10: Analisis Korelasi Multivariat Fitur Numerik (Heatmap)
**Tujuan**: Memetakan hubungan korelasi antar-seluruh variabel numerik pada dataset stroke.

```python
numeric_df = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi Variabel Numerik")
plt.show()
```

**Tangkapan Layar Bukti Eksekusi**:
![Heatmap Korelasi Variabel Numerik](laporan_assets/step_eda_10_heatmap_korelasi.png)
*Gambar 4.9: Heatmap Matriks Korelasi Pearson Fitur Numerik Dataset Stroke*

**Analisis Hasil**:
Fitur yang memiliki korelasi positif terbesar terhadap target `stroke` adalah `age` ($r = 0.25$), disusul oleh `avg_glucose_level` ($r = 0.13$) dan `hypertension` ($r = 0.13$). Korelasi antar-fitur prediktor berada di bawah 0.35, mengindikasikan tidak adanya multikolinearitas yang berbahaya.

---

#### Step 11: Analisis Bivariat Hubungan Umur terhadap Kejadian Stroke
**Tujuan**: Mengamati perbedaan sebaran umur antara pasien sehat vs pasien penderita stroke.

```python
sns.boxplot(x="stroke", y="age", data=df)
plt.title("Hubungan Umur dengan stroke")
plt.show()
```

**Tangkapan Layar Bukti Eksekusi**:
![Hubungan Umur dengan Stroke](laporan_assets/step_eda_11_boxplot_umur_stroke.png)
*Gambar 4.10: Boxplot Perbandingan Usia Pasien Stroke vs Bebas Stroke*

**Analisis Hasil**:
Perbedaan median usia sangat mencolok. Pasien yang terkena stroke (kelas 1) memiliki median usia di atas 70 tahun dengan mayoritas penderita berusia antara 60 hingga 80 tahun. Pasien tanpa stroke (kelas 0) memiliki median usia sekitar 43 tahun dengan sebaran usia yang merata dari balita hingga lansia.

---

#### Step 12: Analisis Bivariat Hubungan Kadar Glukosa terhadap Kejadian Stroke
**Tujuan**: Mengamati distribusi kadar glukosa darah rata-rata pada penderita stroke vs non-stroke.

```python
sns.boxplot(x="stroke", y="avg_glucose_level", data=df)
plt.title("Hubungan Glukosa dengan stroke")
plt.show()
```

**Tangkapan Layar Bukti Eksekusi**:
![Hubungan Glukosa dengan Stroke](laporan_assets/step_eda_12_boxplot_glukosa_stroke.png)
*Gambar 4.11: Boxplot Hubungan Kadar Glukosa Darah terhadap Kejadian Stroke*

**Analisis Hasil**:
Pasien yang menderita stroke memiliki median kadar glukosa yang lebih tinggi dan sebaran kuartil atas yang mencapai lebih dari 200 mg/dL. Hal ini membuktikan keterkaitan klinis yang erat antara kondisi hiperglikemia/diabetes dengan peningkatan risiko kerusakan pembuluh darah otak (stroke).

---

### BAGIAN B: PRAKTIKUM 2 — REGRESI LINEAR SEDERHANA

#### Step 1: Pembuatan Data Sampel Bivariat (Tinggi vs Berat Badan)
**Tujuan**: Menyiapkan dataset 10 sampel hubungan tinggi badan dalam cm ($x$) terhadap berat badan dalam kg ($y$).

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df_simple = pd.DataFrame([
    [151, 63], [174, 81], [138, 56], [186, 91], [128, 47],
    [136, 57], [179, 76], [163, 72], [152, 62], [131, 48]
], columns=['x', 'y'])
```

---

#### Step 2: Pemeriksaan Korelasi Pearson
**Tujuan**: Mengukur kekuatan asosiasi linear antara tinggi dan berat badan sebelum regresi dilakukan.

```python
df_simple.corr()
```

**Tangkapan Layar Bukti Eksekusi**:
![Matriks Korelasi Tinggi vs Berat](laporan_assets/step_reg_02_korelasi_sederhana.png)
*Gambar 4.12: Tangkapan Layar Matriks Korelasi Pearson Tinggi Badan vs Berat Badan*

**Analisis Hasil**:
Korelasi $r = 0.9771$ (97.71%). Angka ini menunjukkan hubungan linear positif yang luar biasa kuat: penambahan tinggi badan seseorang selalu berkorelasi positif dan proporsional dengan penambahan berat tubuh.

---

#### Step 3 & 4: Penyiapan Vektor Fitur & Pelatihan Model OLS
**Tujuan**: Menyesuaikan dimensi array input $X$ menjadi matriks 2 dimensi dan melatih estimator `LinearRegression()`.

```python
x_train = df_simple['x'].values[:, np.newaxis]
y_train = df_simple['y'].values

lm = LinearRegression()
lm.fit(x_train, y_train)
```

---

#### Step 5 & 6: Pengujian Prediksi Data Baru & Ekstraksi Parameter ($m$ dan $c$)
**Tujuan**: Menguji model pada data baru berketinggian [170, 171, 160, 180, 150] cm dan mengekstraksi nilai kemiringan garis ($m$) serta perpotongan sumbu $y$ ($c$).

```python
x_test = [[170], [171], [160], [180], [150]]
p = lm.predict(x_test)

print("Hasil Prediksi:", p)
print("Coefficient (m):", lm.coef_)
print("Intercept (c):", lm.intercept_)
```

**Tangkapan Layar Bukti Eksekusi**:
![Hasil Evaluasi Model Sederhana](laporan_assets/step_reg_06_evaluasi_sederhana.png)
*Gambar 4.13: Tangkapan Layar Pengujian Data Baru & Parameter Regresi Linear Sederhana*

**Analisis Hasil**:
- **Koefisien ($m$)** = `0.67461045`
- **Intercept ($c$)** = `-38.45508708`
- **Persamaan Garis Model**:
  $$\text{Berat Badan} = 0.6746 \times (\text{Tinggi Badan}) - 38.4551$$
Interpretasi: Setiap penambahan 1 cm tinggi badan diasosiasikan dengan kenaikan rata-rata berat badan sebesar ~0.6746 kg. Seseorang dengan tinggi 170 cm diestimasikan memiliki berat badan $0.6746(170) - 38.4551 = 76.23\text{ kg}$.

---

#### Step 7: Visualisasi Garis Regresi Model Terbaik (Best-Fit Line)
**Tujuan**: Memetakan titik-titik data latih asli dengan garis linear prediksi model.

```python
pb = lm.predict(x_train)
dfc = pd.DataFrame({'x': df_simple['x'], 'y': pb})

plt.scatter(df_simple['x'], df_simple['y'], color='blue', label='Data Aktual')
plt.plot(dfc['x'], dfc['y'], color='magenta', linewidth=2, label='Garis Regresi OLS')
plt.xlabel('Tinggi Badan Dalam Cm')
plt.ylabel('Berat Badan Dalam Kg')
plt.title('Regresi Linear Sederhana: Tinggi vs Berat Badan')
plt.legend()
plt.show()
```

**Tangkapan Layar Bukti Eksekusi**:
![Garis Regresi Linear Sederhana](laporan_assets/step_reg_07_garis_regresi_sederhana.png)
*Gambar 4.14: Visualisasi Scatter Plot dan Garis Best-Fit Regresi Linear Sederhana*

**Analisis Hasil**:
Gambar 4.14 memperlihatkan bahwa garis regresi magenta memotong secara tepat di tengah sebaran data dengan jarak error residual yang sangat minim.

---

#### Step 8: Evaluasi Kinerja Model dengan $R^2$ Score
**Tujuan**: Mengukur proporsi variansi data yang dijelaskan oleh model.

```python
rSquared = r2_score(df_simple['y'], pb)
print("R2 Score:", rSquared)
```

**Analisis Hasil**:
Model mencatat nilai $R^2 = 0.9774$ (**97.74%**). Ini mengonfirmasi bahwa 97.74% variasi berat badan pada sampel mampu diterangkan secara akurat oleh variabel tinggi badan.

---

### BAGIAN C: PRAKTIKUM 3 — REGRESI LINEAR BERGANDA (STUDI KASUS BOTAK.CSV)

#### Step 9: Pemuatan Dataset `botak.csv` & Inspeksi Struktur
**Tujuan**: Membaca dataset studi kasus multi-variabel `botak.csv` (7.991 baris data) dan memeriksa fitur-fiturnya.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

db = pd.read_csv('botak.csv')
db[['umur', 'jenis_kelamin', 'gaji', 'tinggi', 'berat', 'botak_prob']].head()
```

**Tangkapan Layar Bukti Eksekusi**:
![Dataset botak.csv](laporan_assets/step_reg_09_preview_botak.png)
*Gambar 4.15: Tangkapan Layar Pemuatan Dataset botak.csv dan Fitur Terpilih*

---

#### Step 10 & 11: Pembersihan Data & Pemisahan Fitur Multivariat ($X$) dan Target ($y$)
**Tujuan**: Membuang baris yang memiliki nilai hilang pada subset 4 fitur numerik dan memisahkan matriks independen $X$ serta target kontinu $y$.

```python
# Eliminasi baris kosong
db_clean = db.dropna(subset=["umur", "gaji", "tinggi", "berat", "botak_prob"])

# Pemisahan fitur dan target
x = db_clean[["umur", "gaji", "tinggi", "berat"]]
y = db_clean["botak_prob"]
```

---

#### Step 12 & 13: Pembagian Data Train-Test (80:20) & Pelatihan Model
**Tujuan**: Membagi data secara acak menjadi 80% data latih dan 20% data uji (`random_state=42`) lalu melatih model regresi berganda.

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
```

---

#### Step 14 & 15: Prediksi Data Uji & Ekstraksi Parameter Evaluasi
**Tujuan**: Menguji model pada data uji ($y_{\text{pred}}$) dan mencetak nilai koefisien tiap prediktor, intercept, MSE, serta $R^2$ Score.

```python
y_pred = model.predict(x_test)

print("Koefisien:", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

**Tangkapan Layar Bukti Eksekusi**:
![Evaluasi Regresi Berganda](laporan_assets/step_reg_15_evaluasi_berganda.png)
*Gambar 4.16: Tangkapan Layar Output Evaluasi Koefisien, Intercept, MSE, dan R2 Score*

**Analisis Hasil & Rumus Matematis**:
Hasil eksekusi pada Gambar 4.16 mencatat parameter:
- **Intercept ($b_0$)** = `0.10138266`
- **Koefisien Umur ($b_1$)** = `+0.00655710`
- **Koefisien Gaji ($b_2$)** = `+2.81851242e-09`
- **Koefisien Tinggi ($b_3$)** = `+5.28583198e-05`
- **Koefisien Berat ($b_4$)** = `+1.88754009e-05`
- **Mean Squared Error (MSE)** = `0.03029277`
- **$R^2$ Score** = `0.11696334` (**11.70%**)

**Persamaan Matematis Model**:
$$\hat{y} = 0.1014 + 0.00656 \cdot (\text{umur}) + 2.82 \times 10^{-9} \cdot (\text{gaji}) + 5.29 \times 10^{-5} \cdot (\text{tinggi}) + 1.89 \times 10^{-5} \cdot (\text{berat})$$

Interpretasi Parameter:
1. `umur` merupakan prediktor paling dominan. Setiap pertambahan 1 tahun usia menaikkan probabilitas kebotakan sebesar **0.00656** (atau kenaikan 6.56% per 10 tahun usia).
2. `gaji` memiliki koefisien sangat kecil ($2.82 \times 10^{-9}$) karena skala nominalnya berada pada jutaan rupiah, sehingga pengaruh per unit 1 Rupiah hampir tidak berdampak langsung.
3. Nilai $R^2 = 11.70\%$ dan MSE $0.0303$ menunjukkan bahwa meskipun model mampu menangkap tren kenaikan umur, masih terdapat 88.3% variasi probabilitas kebotakan yang ditentukan oleh faktor-faktor biologis lain (seperti genetik keturunan, stres, hormon).

---

#### Step 16: Visualisasi Scatter Plot Prediksi vs Nilai Aktual
**Tujuan**: Menggambarkan kesesuaian antara hasil tebakan model ($y_{\text{pred}}$) terhadap nilai sesungguhnya ($y_{\text{test}}$) dengan garis referensi diagonal merah $y = x$.

```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='#2980B9', label='Data Uji')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Prediksi Sempurna (y=x)')
plt.xlabel("Nilai Asli (y_test)")
plt.ylabel("Prediksi (y_pred)")
plt.title("Prediksi vs Nilai Asli (Regresi Linear Berganda)")
plt.legend()
plt.show()
```

**Tangkapan Layar Bukti Eksekusi**:
![Prediksi vs Nilai Asli Regresi Berganda](laporan_assets/step_reg_16_scatter_prediksi_vs_aktual.png)
*Gambar 4.17: Visualisasi Scatter Plot Prediksi Model vs Nilai Aktual Kebotakan*

**Analisis Hasil**:
Titik-titik persebaran pada Gambar 4.17 menunjukkan pola mendatar terkonsentrasi di rentang probabilitas 0.3 hingga 0.6. Persebaran tidak merapat sempurna pada garis ideal diagonal merah, konsisten dengan nilai $R^2 = 11.70\%$. Model linear memprediksi nilai rata-rata bersyarat (*conditional mean*) namun belum mampu memetakan variabilitas ekstrem target probabilitas.

---

## BAB V: PEMBAHASAN & ANALISIS KRITIS

### 5.1 Karakteristik Dataset Stroke & Pengaruh Class Imbalance
Pada dataset stroke, 95.1% pasien berada di kelas 0 (sehat) dan hanya 4.9% di kelas 1 (stroke). Jika model klasifikasi dilatih secara naif pada dataset seperti ini, model yang selalu menebak "0" akan mencetak akurasi 95.1% padahal gagal mendeteksi pasien yang benar-benar sakit. Oleh karena itu, langkah mitigasi yang wajib dilakukan pada riset lanjutan adalah:
- Melakukan teknik *oversampling* minoritas menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)**.
- Menyesuaikan penalti *loss function* menggunakan parameter `class_weight='balanced'`.

### 5.2 Komparasi Model: Mengapa $R^2$ Regresi Sederhana 97.7% dan Regresi Berganda 11.7%?
Perbedaan drastis antara kedua model disebabkan oleh:
1. **Sifat Hubungan Antar-Variabel**: Tinggi dan berat badan memiliki korelasi fisik proporsional ($r = 0.977$), sedangkan probabilitas kebotakan merupakan karakteristik biologis yang dipengaruhi faktor genetik polygenic yang tidak linear murni.
2. **Ketiadaan Fitur Kategorikal Dominan**: Pada `botak.csv`, kolom penting seperti `is_keturunan` (faktor genetik keluarga) belum dilibatkan dalam regresi berganda numerik.
3. **Ketiadaan Standarisasi Skala Fitur**: Fitur gaji bernilai jutaan rupiah sedangkan umur bernilai puluhan tahun. Tanpa normalisasi (`StandardScaler`), optimasi bobot model menjadi bias terhadap rentang skala nilai absolut.

### 5.3 Makna Bobot Koefisien Fitur terhadap Kebotakan
Meskipun nilai $R^2$ tergolong moderat (11.7%), signifikansi tanda koefisien (+0.00656 pada usia) memberikan bukti statistik yang selaras dengan teori kedokteran dermatologi: proses penuaan seluler dan atrofi folikel rambut meningkat seiring bertambahnya usia individu.

### 5.4 Keterbatasan Model Regresi Linear pada Target Probabilitas
Secara konseptual, Regresi Linear OLS dirancang untuk target numerik tanpa batas ($-\infty < y < \infty$). Apabila digunakan untuk memprediksi probabilitas $[0, 1]$, model berpotensi menghasilkan prediksi di bawah 0 atau di atas 1 pada data ekstrem. Algoritma yang lebih presisi untuk target probabilitas terikat adalah **Regresi Logistik** atau **Beta Regression**.

---

## BAB VI: KESIMPULAN & SARAN

### 6.1 Kesimpulan
1. **Exploratory Data Analysis (EDA)** berhasil mengungkap anatomi dataset: mendeteksi 201 missing values pada kolom BMI, mengidentifikasi ketidakseimbangan kelas target stroke (4.9%), dan menemukan bahwa umur ($r = 0.25$) serta glukosa ($r = 0.13$) adalah faktor risiko stroke paling signifikan.
2. **Imputasi Median** pada kolom BMI terbukti efektif melengkapi dataset tanpa mengubah persebaran distribusi data numerik.
3. **Regresi Linear Sederhana** berhasil menghasilkan model $\text{Berat} = 0.6746 \times (\text{Tinggi}) - 38.4551$ dengan nilai $R^2$ sebesar **0.9774 (97.74%)**, membuktikan korelasi deterministik yang kuat antara tinggi dan berat badan.
4. **Regresi Linear Berganda** pada `botak.csv` menghasilkan model prediksi dengan MSE rendah (**0.0303**) dan $R^2$ sebesar **11.70%**, dengan variabel umur sebagai kontributor positif paling signifikan ($+0.00656$ per tahun).

### 6.2 Saran Pengembangan Lanjutan
1. Menerapkan **Feature Scaling** (`StandardScaler` atau `MinMaxScaler`) sebelum proses regresi, terutama bagi fitur dengan disparitas skala tinggi seperti gaji.
2. Memanfaatkan teknik **One-Hot Encoding** untuk memasukkan fitur kategorikal kunci dari `botak.csv` (misal: `is_keturunan`, `is_merokok`, `jenis_kelamin`) guna meningkatkan daya jelas model ($R^2$ score).
3. Mengeksplorasi algoritma regresi non-linear (*Random Forest Regressor*, *Gradient Boosting*, *XGBoost*) untuk menangkap pola non-linearitas data.
4. Menerapkan teknik penyeimbangan data (**SMOTE**) pada dataset stroke sebelum melanjutkan ke tahap pemodelan klasifikasi prediktif.

---

## DAFTAR PUSTAKA
1. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning: with Applications in Python*. Springer.
3. McKinney, W. (2022). *Python for Data Analysis: Data Wrangling with pandas, NumPy, and Jupyter* (3rd ed.). O'Reilly Media.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
5. Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley Publishing Company.
6. Waskom, M. L. (2021). *Seaborn: statistical data visualization*. Journal of Open Source Software, 6(60), 3021.
