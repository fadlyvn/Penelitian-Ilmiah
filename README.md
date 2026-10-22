# Analisis Sentimen Ulasan Aplikasi Amazon

Project analisis sentimen terhadap ulasan pengguna aplikasi Amazon (Google Play Store). Ulasan berbahasa Inggris diproses lewat tahapan text preprocessing, dilabeli sentimen **Positive**, **Negative**, dan **Neutral** menggunakan leksikon AFINN, lalu diklasifikasikan dengan **Support Vector Machine (SVM)** berbasis fitur **TF-IDF**.

## Daftar Isi

- [Ringkasan](#ringkasan)
- [Dataset](#dataset)
- [Teknologi](#teknologi)
- [Alur Pengerjaan](#alur-pengerjaan)
- [Hasil](#hasil)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Struktur Folder](#struktur-folder)
- [Keterbatasan](#keterbatasan)
- [Lisensi](#lisensi)

## Ringkasan

| Item | Keterangan |
|------|------------|
| Tujuan | Mengklasifikasikan sentimen ulasan pengguna aplikasi Amazon |
| Bahasa data | Inggris |
| Pelabelan | Berbasis leksikon (AFINN) |
| Fitur | TF-IDF |
| Model | SVM (hyperparameter dicari dengan `RandomizedSearchCV`) |
| Penanganan imbalance | SMOTE |
| Akurasi (cross-validation 5-fold) | ± 81,2% |

## Dataset

- **File:** `amazon_reviews.csv` (ulasan aplikasi Amazon dari Google Play Store)
- **Kolom yang dipakai:** `userName`, `content` (teks ulasan), `score` (rating)
- **Sampling:** 15.000 baris diambil acak (`random_state=42`)
- **Setelah pembersihan:** 14.631 ulasan (data kosong dan duplikat pada `content` dihapus)
- **Pembagian data:** 80% latih (11.704) dan 20% uji (2.927)

Letakkan file CSV di folder `data/`, lalu sesuaikan path pada sel pemuatan data di notebook (di notebook asli path-nya `/content/amazon_reviews.csv`, yaitu path Google Colab).

## Teknologi

- Python 3
- pandas, numpy
- NLTK (tokenisasi, stopwords, Porter Stemmer)
- AFINN (pelabelan sentimen)
- scikit-learn (TF-IDF, SVM, RandomizedSearchCV, metrik evaluasi)
- imbalanced-learn (SMOTE)
- matplotlib, seaborn, plotly, wordcloud (visualisasi)
- Jupyter Notebook / Google Colab

## Alur Pengerjaan

1. **Load data**: membaca CSV dan mengambil sampel 15.000 ulasan.
2. **Pembersihan data**: menghapus nilai kosong dan duplikat, mengubah ke huruf kecil, menghapus karakter selain huruf, serta merapikan spasi.
3. **Tokenisasi**: memecah teks menjadi token dengan `nltk.word_tokenize`.
4. **Stopword removal**: menghapus stopwords bahasa Inggris dari NLTK.
5. **Stemming**: menggunakan `PorterStemmer`.
6. **Pelabelan**: skor sentimen dihitung dengan AFINN pada teks yang sudah dibersihkan.
   - skor > 0 → **Positive**
   - skor < 0 → **Negative**
   - skor = 0 → **Neutral**
7. **TF-IDF**: mengubah teks menjadi vektor fitur dengan `TfidfVectorizer` (14.896 fitur).
8. **Visualisasi**: word cloud per kelas sentimen, funnel chart, dan countplot distribusi label.
9. **Split data**: `train_test_split` dengan `test_size=0.2`.
10. **SMOTE**: oversampling pada data latih untuk menyeimbangkan kelas.
11. **Pemodelan SVM**: `RandomizedSearchCV` (`n_iter=10`, `cv=5`) dengan ruang pencarian:
    - `C`: 0.1, 1, 10
    - `kernel`: linear, rbf, poly
    - `gamma`: scale, auto
12. **Evaluasi**: cross-validation 5-fold, confusion matrix, dan classification report.

**Parameter terbaik:** `kernel='linear'`, `C=10`, `gamma='auto'`

## Hasil

**Cross-validation 5-fold (seluruh data):**

| Metrik | Nilai |
|--------|-------|
| Accuracy | 0,812 |
| Precision (weighted) | 0,816 |
| Recall (weighted) | 0,812 |
| F1-score (weighted) | 0,814 |

**Classification report pada data uji:**

| Kelas | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0,84 | 0,81 | 0,83 | 1.132 |
| Neutral | 0,53 | 0,64 | 0,58 | 481 |
| Positive | 0,91 | 0,86 | 0,89 | 1.314 |
| **Accuracy** | | | **0,81** | 2.927 |
| Macro avg | 0,76 | 0,77 | 0,76 | 2.927 |
| Weighted avg | 0,82 | 0,81 | 0,81 | 2.927 |

Model paling baik mengenali kelas **Positive**, sedangkan kelas **Neutral** paling sulit diprediksi.

Tambahkan gambar hasil visualisasi dari notebook (confusion matrix, word cloud) ke folder `images/` lalu tampilkan di sini, misalnya:

```markdown
![Confusion Matrix](images/confusion_matrix.png)
![Word Cloud Positif](images/wordcloud_positif.png)
```

## Instalasi

1. Clone repository:
   ```bash
   git clone https://github.com/fadlyvn/Penelitian-Ilmiah.git
   cd fadlyvn/Penelitian-Ilmiah
   ```

2. (Opsional) Buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```

4. Unduh resource NLTK (dijalankan otomatis di notebook, atau manual):
   ```python
   import nltk
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   ```

## Cara Menjalankan

1. Taruh dataset `amazon_reviews.csv` di folder `data/` dan sesuaikan path-nya di notebook.
2. Jalankan notebook:
   ```bash
   jupyter notebook PI.ipynb
   ```
3. Jalankan semua sel secara berurutan dari atas ke bawah.

Notebook juga bisa dibuka langsung di Google Colab. Unggah `amazon_reviews.csv` ke sesi Colab agar path `/content/amazon_reviews.csv` berfungsi.

## Struktur Folder

```
.
├── data/
│   └── amazon_reviews.csv
├── images/                 # (opsional) gambar hasil visualisasi
├── PI.ipynb                # Notebook utama
├── requirements.txt
└── README.md
```

## Keterbatasan

- **Label berasal dari leksikon AFINN, bukan anotasi manusia.** Model SVM sebenarnya mempelajari cara kerja AFINN, sehingga akurasi 81% mengukur seberapa baik model meniru AFINN, bukan seberapa benar sentimen aslinya.
- Kolom `score` (rating bintang) tidak dipakai sebagai label. Ulasan dengan rating tinggi bisa saja berisi kritik, dan sebaliknya.
- Kelas **Neutral** punya performa terendah (F1 0,58), sebagian karena skor AFINN 0 juga mencakup ulasan yang tidak mengandung kata berlabel sama sekali.
- Stopword removal menghapus kata seperti *not* dan *no*, sehingga negasi tidak tertangkap. Hal ini tidak memengaruhi label (AFINN dihitung sebelum tokenisasi), tetapi bisa memengaruhi pemahaman model terhadap konteks.
- Hanya 15.000 ulasan yang dipakai, dan hanya berbahasa Inggris.
- Pada notebook, TF-IDF dilatih di seluruh data sebelum split, sehingga ada potensi kebocoran informasi kecil dari data uji.




