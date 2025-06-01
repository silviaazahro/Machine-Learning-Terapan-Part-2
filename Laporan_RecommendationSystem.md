# Laporan Proyek Machine Learning - Collaborative Filtering pada Dataset Buku

## 📌 Project Overview

Dalam era digital yang semakin berkembang, sistem rekomendasi memiliki peranan penting dalam membantu pengguna menavigasi informasi dan pilihan produk yang sangat besar. Salah satu aplikasi populer dari sistem rekomendasi adalah pada layanan e-commerce atau platform penyedia buku. Dengan semakin banyaknya pilihan buku dan meningkatnya kebutuhan personalisasi pengalaman pengguna, sistem rekomendasi yang efektif menjadi solusi penting untuk meningkatkan _engagement_ dan kepuasan pengguna.

Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis _collaborative filtering_ menggunakan dataset buku yang berisi informasi metadata seperti judul, penulis, tahun terbit, jumlah halaman, rating, dan lain-lain. Fokus utama proyek ini adalah memprediksi buku yang kemungkinan besar akan disukai oleh pengguna berdasarkan kesamaan perilaku atau preferensi pengguna lain.

Dataset ini bersifat publik dan mencakup berbagai atribut penting, di antaranya:

- `title`: Judul buku
- `authors`: Penulis
- `published_year`: Tahun terbit
- `average_rating`: Rata-rata rating pengguna
- `num_pages`: Jumlah halaman
- `ratings_count`: Jumlah pengguna yang memberi rating

Teknik _collaborative filtering_ yang digunakan dalam proyek ini berbasis pada pendekatan **neural network embedding**, di mana fitur-fitur numerik dari pengguna dan item (dalam hal ini: tahun terbit dan jumlah halaman sebagai proxy untuk pengguna dan item) dipetakan ke dalam vektor laten untuk mempelajari pola interaksi.

> 📚 Penelitian terdahulu menunjukkan bahwa pendekatan matrix factorization dan deep learning dapat memberikan hasil yang baik dalam membangun sistem rekomendasi, di antaranya:
>
> - Koren, Bell, & Volinsky (2009) — ["Matrix Factorization Techniques for Recommender Systems"](https://ieeexplore.ieee.org/abstract/document/5197422)
> - Covington et al. (2016) — ["Deep Neural Networks for YouTube Recommendations"](https://dl.acm.org/doi/abs/10.1145/2959100.2959190)

Dalam proyek ini, dua model _collaborative filtering_ yang dikembangkan adalah:

1. **RecommenderNet** — Model sederhana berbasis embedding dan _dot product_ antara fitur pengguna dan item.
2. **NeuMF (Neural Matrix Factorization)** — Kombinasi antara pendekatan GMF (_Generalized Matrix Factorization_) dan MLP (_Multilayer Perceptron_) yang lebih kompleks.

Model-model tersebut dilatih menggunakan data interaksi buatan antara `published_year` dan `num_pages` untuk mempelajari hubungan laten antar entitas, dan menghasilkan rekomendasi buku berdasarkan prediksi skor atau rating.

## 🧠 Business Understanding
---

### ❓ Problem Statements

- Bagaimana memprediksi rating buku yang belum pernah dibaca oleh pengguna?
- Bagaimana memberikan rekomendasi buku yang relevan dan dipersonalisasi untuk pengguna berdasarkan histori dan preferensi mereka?

### 🎯 Goals

- Mengembangkan model **collaborative filtering** untuk memprediksi rating buku yang belum pernah dibaca oleh pengguna.
- Menghasilkan **top-N rekomendasi buku** yang disesuaikan dengan karakteristik pengguna (dalam hal ini representasi waktu dan fitur konten buku seperti jumlah halaman).

### 💡 Solution Statements

- Menggunakan pendekatan **matrix factorization** melalui embedding fitur pengguna dan item untuk mempelajari representasi laten (_latent representations_).
- Mengimplementasikan dua pendekatan sistem rekomendasi:
  1. **Dot product** antara embedding pengguna (`published_year`) dan item (`num_pages`) untuk memprediksi rating — mendekati baseline matrix factorization.
  2. **Neural Collaborative Filtering (NeuMF)** — model yang menggabungkan _Generalized Matrix Factorization (GMF)_ dan _Multi-Layer Perceptron (MLP)_ untuk meningkatkan performa prediksi.
- Menggunakan **TensorFlow dan Keras** sebagai framework utama dalam membangun dan melatih model.

## 📊 Data Understanding
---

Dataset yang digunakan dalam proyek ini adalah **Books Dataset** yang tersedia secara publik melalui [Kaggle](https://www.kaggle.com/datasets/abdallahwagih/books-dataset). Dataset ini mencakup informasi bibliografis dan metrik evaluasi terhadap berbagai judul buku, dan cocok digunakan dalam proyek sistem rekomendasi berbasis konten maupun kolaboratif. Dataset berjumlah 12 kolom, 6810 baris dan dataset kotor yang masih terdapat missing values pada kolom subtitle, authors, categories, thumbnail, description, description, published_year, average_rating, num_pages, ratings_count, dataset ini diambil dari platform

Dataset terdiri dari satu file utama:

- `dataset_buku.csv`: berisi informasi deskriptif mengenai ribuan buku, seperti ISBN, judul, penulis, kategori, tahun terbit, jumlah halaman, rating rata-rata, dan jumlah rating.

### 🔢 Fitur-fitur dalam `dataset_buku.csv`:

- `isbn13`: ID unik buku versi 13 digit.
- `isbn10`: ID unik buku versi 10 digit.
- `title`: Judul utama buku.
- `subtitle`: Subjudul (jika tersedia).
- `authors`: Nama penulis.
- `categories`: Kategori atau genre buku.
- `thumbnail`: URL gambar sampul.
- `description`: Deskripsi buku.
- `published_year`: Tahun terbit buku.
- `average_rating`: Rating rata-rata buku (skala 0–5).
- `num_pages`: Jumlah halaman buku.
- `ratings_count`: Jumlah rating yang diterima buku.

### 🧹 Kondisi Data

Dataset ini memiliki beberapa nilai yang hilang (*missing values*) pada kolom-kolom berikut:

| Kolom            | Jumlah Missing Value |
|------------------|----------------------|
| subtitle         | 4.429                |
| authors          | 72                   |
| categories       | 99                   |
| thumbnail        | 329                  |
| description      | 262                  |
| published_year   | 6                    |
| average_rating   | 43                   |
| num_pages        | 43                   |
| ratings_count    | 43                   |

> Kolom `isbn13`, `isbn10`, dan `title` tidak memiliki nilai yang hilang.

### 🧭 Distribusi Data

- `published_year` memiliki rentang dari **1853** hingga **2019**, mencakup lebih dari satu abad penerbitan buku.
- `num_pages` berkisar dari puluhan hingga lebih dari seribu halaman, mencerminkan variasi panjang buku yang sangat besar.
- `average_rating` menunjukkan kecenderungan distribusi skewed ke arah nilai tinggi, mirip dengan dataset rating pada sistem rekomendasi lainnya.
- `ratings_count` sangat bervariasi; beberapa buku mendapatkan ribuan rating, sementara sebagian lainnya hanya beberapa.

### 📌 Catatan

Karena tidak tersedia informasi `userId` eksplisit, pendekatan simulatif digunakan untuk membangun sistem rekomendasi, di mana `published_year` diasumsikan mewakili sekelompok pembaca dari generasi yang sama, dan digunakan sebagai pengganti `userId` dalam konteks model rekomendasi berbasis collaborative filtering.

### 📊 Visualisasi Distribusi Data Rating

Grafik batang di bawah ini menunjukkan frekuensi masing-masing nilai rating buku dari skala 0 hingga 5. Terlihat bahwa rating **4.0** merupakan yang paling sering muncul, diikuti oleh rating **3.0** dan **5.0**. Hal ini mengindikasikan bahwa sebagian besar pembaca memberikan penilaian yang cukup positif terhadap buku-buku yang mereka baca.

![Distribusi Rating](https://github.com/silviaazahro/Machine-Learning-Terapan-Part-2/blob/main/Distribusi%20Rating.png)

---

### 🧱 Visualisasi Histogram Data Rating

Histogram di bawah ini menggambarkan penyebaran nilai rating dalam bentuk batang berwarna biru muda. Pola distribusinya mirip dengan grafik distribusi sebelumnya, di mana rating tinggi (khususnya 4.0) mendominasi. Ini menunjukkan bahwa persepsi pembaca terhadap buku dalam dataset ini cenderung positif.

![Histogram Rating](https://github.com/silviaazahro/Machine-Learning-Terapan-Part-2/blob/main/Histogram%20Rating.png)

---

### 📦 Visualisasi Boxplot Data Rating

Boxplot berikut menampilkan ringkasan statistik nilai rating buku, termasuk nilai minimum, maksimum, kuartil, dan median. Median rating berada di sekitar **3.8**, menunjukkan bahwa mayoritas rating berkisar pada nilai tinggi. Beberapa outlier dengan rating rendah juga terlihat, mencerminkan adanya buku yang dianggap kurang memuaskan oleh sebagian pembaca.

![Boxplot Rating](https://github.com/silviaazahro/Machine-Learning-Terapan-Part-2/blob/main/Boxplot%20Rating.png)

## Data Preparation 
---
Beberapa tahapan dalam persiapan data yang dilakukan meliputi:

1. **Penanganan Missing Value:**
        * Teknik: Imputasi dengan nilai mean dan median.
        * Kode Snippet:
            ```python
   data['subtitle'] = data['subtitle'].fillna('')
   data['authors'] = data['authors'].fillna('Unknown')
   data['categories'] = data['categories'].fillna('Other')
   data['thumbnail'] = data['thumbnail'].fillna('https://example.com/default-thumbnail.jpg')
   data['description'] = data['description'].fillna('No description available')
   data['published_year'] = data['published_year'].fillna(data['published_year'].median())
   data['average_rating'] = data['average_rating'].fillna(data['average_rating'].mean())
   data['num_pages'] = data['num_pages'].fillna(data['num_pages'].median())
   data['ratings_count'] = data['ratings_count'].fillna(0)
            ```
        * Proses: Missing value pada fitur `bmi` diisi dengan nilai median dari fitur tersebut.
        * Alasan: Fitur `bmi` memiliki missing value yang cukup signifikan. Imputasi dengan median dipilih karena mean dan median robust terhadap outlier, yang mungkin ada dalam distribusi `bmi`. Hal ini mencegah outlier mendistorsi representasi tipikal dari data.
3. **Encoding**: Mengonversi variabel *published_year* dan *num_pages* menjadi indeks bilangan bulat agar dapat dimanfaatkan dalam layer embedding. Layer embedding membutuhkan input berupa indeks numerik yang nantinya dipetakan ke dalam vektor representasi laten.  
4. **Normalisasi**: Melakukan normalisasi pada data *rating* ke dalam rentang 0 hingga 1 karena model menggunakan fungsi aktivasi sigmoid di output layer. Hal ini penting agar output model sesuai dengan skala target yang diinginkan.  
5. **Pembagian Dataset**: Data dipisah menjadi set pelatihan (*training set*) dan set validasi (*validation set*) untuk mengukur performa model secara objektif serta menghindari overfitting.

Tahapan ini dilakukan agar data siap digunakan secara efektif dalam model deep learning.

## Modeling
---

Dalam proyek ini, dikembangkan dua jenis model rekomendasi berbasis collaborative filtering. Kedua pendekatan tersebut digunakan untuk membandingkan kinerja model klasik yang memakai matrix factorization dengan model deep learning yang menggunakan neural network, dengan tujuan mengevaluasi performa serta fleksibilitas masing-masing dalam konteks rekomendasi buku.

### 1. **Matrix Factorization dengan Embedding (Baseline)**

Model ini menerapkan pendekatan collaborative filtering klasik yang diimplementasikan dengan menggunakan layer embedding pada TensorFlow.

**Arsitektur:**
- Terdapat dua embedding layer, masing-masing untuk *year* dan *pages*.  
- Skor kecocokan dihitung melalui operasi **dot product** antar embedding tersebut.  
- Ditambahkan bias dan fungsi aktivasi sigmoid agar output berada pada rentang 0 hingga 1.

     ```python
     class RecommenderNet(tf.keras.Model):
    def __init__(self, num_year, num_pages, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_year = num_year
        self.num_pages = num_pages
        self.embedding_size = embedding_size

        # Embedding untuk year
        self.year_embedding = layers.Embedding(
            num_year,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.year_bias = layers.Embedding(num_year, 1)

        # Embedding untuk pages
        self.pages_embedding = layers.Embedding(
            num_pages,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.pages_bias = layers.Embedding(num_pages, 1)

    def call(self, inputs):
        year_vector = self.year_embedding(inputs[:, 0])
        year_bias = self.year_bias(inputs[:, 0])
        pages_vector = self.pages_embedding(inputs[:, 1])
        pages_bias = self.pages_bias(inputs[:, 1])

        # Dot product antara year dan pages embedding
        dot_year_pages = tf.reduce_sum(year_vector * pages_vector, axis=1, keepdims=True)

        # Menambahkan bias
        x = dot_year_pages + year_bias + pages_bias

        # Aktivasi sigmoid untuk output antara 0 dan 1
        return tf.nn.sigmoid(x)
     ```

**Keunggulan:**  
- Struktur model yang sederhana dan efisien.  
- Proses pelatihan berlangsung dengan cepat.  
- Ideal digunakan sebagai baseline atau model awal.

**Keterbatasan:**  
- Hubungan antara *year* dan *pages* hanya bersifat linear.  
- Kurang mampu menangkap pola kompleks atau non-linear dalam preferensi pengguna.

### 2. **Neural Matrix Factorization (NeuMF)**

NeuMF adalah pendekatan yang mengkombinasikan dua jalur, yaitu Generalized Matrix Factorization (GMF) dan Multi-Layer Perceptron (MLP). Metode ini lebih fleksibel karena mampu menangkap hubungan non-linear antar embedding.

**Arsitektur:**
- Embedding untuk year dan pages diproses melalui dua jalur berbeda:
  - **GMF**: menggunakan operasi dot product seperti pada model klasik.  
  - **MLP**: menggabungkan (concatenate) embedding kemudian melewati beberapa fully connected layer.  
- Output dari kedua jalur tersebut digabungkan dan diteruskan ke dense layer terakhir.  
- Fungsi aktivasi sigmoid digunakan pada output untuk menghasilkan skor prediksi.

     ```python
  from tensorflow.keras import Input, Model, layers
import tensorflow as tf

def get_NeuMF_model(num_years, num_pages, mf_dim=8, mlp_layers=[64,32,16,8], dropout=0.0):
    # Input layer
    year_input = Input(shape=(1,), name="year_input")
    page_input = Input(shape=(1,), name="page_input")

    # MF part embedding
    mf_year_embedding = layers.Embedding(num_years, mf_dim, name="mf_year_embedding")(year_input)
    mf_page_embedding = layers.Embedding(num_pages, mf_dim, name="mf_page_embedding")(page_input)
    mf_year_embedding = layers.Flatten()(mf_year_embedding)
    mf_page_embedding = layers.Flatten()(mf_page_embedding)
    mf_vector = layers.multiply([mf_year_embedding, mf_page_embedding])

    # MLP part embedding
    mlp_embedding_dim = mlp_layers[0] // 2
    mlp_year_embedding = layers.Embedding(num_years, mlp_embedding_dim, name="mlp_year_embedding")(year_input)
    mlp_page_embedding = layers.Embedding(num_pages, mlp_embedding_dim, name="mlp_page_embedding")(page_input)
    mlp_year_embedding = layers.Flatten()(mlp_year_embedding)
    mlp_page_embedding = layers.Flatten()(mlp_page_embedding)
    mlp_vector = layers.concatenate([mlp_year_embedding, mlp_page_embedding])

    # MLP layers
    for idx, units in enumerate(mlp_layers[1:]):
        mlp_vector = layers.Dense(units, activation='relu', name=f"mlp_dense_{idx}")(mlp_vector)
        if dropout > 0:
            mlp_vector = layers.Dropout(dropout)(mlp_vector)

    # Concatenate MF and MLP parts
    neumf_vector = layers.concatenate([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = layers.Dense(1, activation="sigmoid", name="prediction")(neumf_vector)

    model = Model(inputs=[year_input, page_input], outputs=prediction)
    return model
     ```

**Keunggulan:**  
- Mampu menangkap pola interaksi yang lebih rumit dan kompleks.  
- Lebih fleksibel dengan memanfaatkan kemampuan neural network untuk belajar hubungan non-linear.

**Keterbatasan:**  
- Struktur model lebih kompleks dan proses pelatihan lebih lambat dibandingkan model klasik.  
- Memerlukan penyesuaian hyperparameter yang lebih teliti dan cermat.

### Rekomendasi Top-N dari Dataset Buku

Dari hasil perbandingan kedua model rekomendasi buku, terlihat bahwa keduanya sama-sama merekomendasikan beberapa judul populer dan berkualitas, seperti *Harry Potter and the Sorcerer’s Stone*, *Pride and Prejudice*, dan *To Kill a Mockingbird*, yang menunjukkan konsistensi dalam mengenali buku-buku favorit pembaca. Namun, **NeuMF** cenderung merekomendasikan lebih banyak buku dengan tema klasik dan literatur mendalam seperti *Crime and Punishment* dan *The Great Gatsby*, sementara model klasik berbasis matrix factorization lebih condong pada buku-buku populer dan best-seller seperti *The Hunger Games* dan *The Da Vinci Code*. Hal ini mengindikasikan bahwa NeuMF mampu menangkap preferensi yang lebih kompleks dan spesifik, sedangkan model klasik memberikan rekomendasi yang lebih umum dan mainstream, sesuai dengan karakteristik masing-masing arsitektur.

Meskipun NeuMF menghasilkan rekomendasi yang lebih beragam dan “unik” dalam hal genre dan tema, hasil evaluasi kuantitatif seperti RMSE, MAE, dan R² menunjukkan bahwa model klasik matrix factorization memiliki performa prediksi rating yang sedikit lebih baik. Ini mengimplikasikan bahwa walaupun NeuMF mampu mengeksplorasi preferensi pengguna secara lebih mendalam, ketepatan prediksi rating aktual masih lebih unggul pada model klasik. Oleh karena itu, dalam konteks sistem rekomendasi berbasis prediksi rating buku, model matrix factorization tetap menjadi pilihan yang lebih direkomendasikan.

| Rank | **RecommenderNet**                                    | Predicted Score | Predicted Rating | **NeuMF**                                  | Predicted Score | Predicted Rating |
|-------|------------------------------------------------------|-----------------|------------------|--------------------------------------------|-----------------|------------------|
| 1     | The Collected Letters of C.S. Lewis, Volume 1        | 0.7939          | 3.9693           | The Complete Short Stories of Mark Twain   | 0.7918          | 3.9589           |
| 2     | Democracy in America                                  | 0.7801          | 3.9006           | Roughing It                                | 0.7888          | 3.9440           |
| 3     | The Autobiography of Mark Twain                      | 0.7714          | 3.8569           | The Princess of the Chalet School          | 0.7884          | 3.9419           |
| 4     | The Return of the King                                | 0.7659          | 3.8296           | I Am that                                  | 0.7882          | 3.9412           |
| 5     | Keats's Poetry and Prose                              | 0.7619          | 3.8095           | Selected Letters, 1957-1969                 | 0.7882          | 3.9409           |
| 6     | Cross Stitch                                         | 0.7598          | 3.7988           | Paradise                                   | 0.7872          | 3.9360           |
| 7     | Three Complete Novels                                | 0.7579          | 3.7896           | The Mutineer                               | 0.7837          | 3.9183           |
| 8     | Judas Unchained                                     | 0.7579          | 3.7896           | Music & Silence                            | 0.7836          | 3.9179           |
| 9     | Ludwig Wittgenstein                                 | 0.7576          | 3.7880           | Terre                                      | 0.7831          | 3.9155           |
| 10    | Rain of Gold                                        | 0.7551          | 3.7756           | Democracy in America                        | 0.7831          | 3.9153           |

## Evaluation
---

Tujuan utama dari sistem rekomendasi ini adalah untuk memprediksi **rating** yang kemungkinan besar akan diberikan oleh pengguna terhadap sebuah buku dengan tingkat akurasi yang tinggi. Untuk menilai seberapa baik performa model dalam melakukan prediksi tersebut, digunakan beberapa metrik regresi yang mampu mengukur kedekatan antara hasil prediksi dan nilai sebenarnya. Berikut adalah metrik evaluasi yang digunakan:

### 1. Root Mean Squared Error (RMSE)
**RMSE** menghitung akar dari rata-rata kuadrat selisih antara rating sebenarnya ($y_i$) dan hasil prediksi ($\hat{y}_i$). Nilai **RMSE** yang rendah menunjukkan bahwa model mampu memberikan prediksi yang dekat dengan nilai aktual. Karena menggunakan kuadrat dalam perhitungannya, metrik ini cukup sensitif terhadap nilai ekstrem (outlier). Dalam konteks rekomendasi berbasis rating seperti pada dataset buku, **RMSE** sering dijadikan acuan utama karena penalti terhadap kesalahan besar sangat diperhatikan.

**Rumus**:
  $$
  RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }
  $$

### 2. Mean Absolute Error (MAE)
**MAE** mengukur rata-rata dari perbedaan absolut antara nilai aktual dan prediksi. Berbeda dengan RMSE, **MAE** tidak terlalu terpengaruh oleh outlier, sehingga memberikan gambaran rata-rata kesalahan yang lebih stabil dan moderat. Metrik ini bermanfaat untuk melihat sejauh mana model cenderung “melenceng” dari nilai sebenarnya secara umum.

**Rumus**:
  $$
  MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
  $$

### 3. R-squared (R² Score)
**R²** atau koefisien determinasi menunjukkan seberapa besar variabilitas dari data aktual yang dapat dijelaskan oleh model. Nilainya berkisar:
- **1.0** → prediksi sangat akurat (sempurna)
- **0.0** → model tidak lebih baik dari sekadar rata-rata
- **Kurang dari 0** → performa model lebih buruk daripada tebakan acak

Walaupun **R²** bukan metrik utama dalam sistem rekomendasi, penggunaannya bisa membantu memberikan perspektif tambahan terhadap seberapa baik model menjelaskan variasi data.

**Rumus**:
  $$
  R^2 = 1 - \frac{ \sum (y_i - \hat{y}_i)^2 }{ \sum (y_i - \bar{y})^2 }
  $$

## 📊 Hasil Evaluasi Metrik

| Metrik  | RecommenderNet | NeuMF    |
|---------|----------------|----------|
| RMSE    | 0.450705         | 0.346713   |
| MAE     | 0.319360         | 0.242892   |
| R²      | -0.757065        | -0.039785  |

**Analisis:**
- Model **NeuMF** menunjukkan performa **lebih unggul** dibandingkan RecommenderNet berdasarkan ketiga metrik evaluasi.
- Nilai **RMSE** (Root Mean Squared Error) dan **MAE** (Mean Absolute Error) yang lebih rendah pada NeuMF menunjukkan bahwa model ini mampu memprediksi rating dengan tingkat kesalahan yang lebih kecil.
- Walaupun nilai **R²** pada kedua model negatif (menunjukkan bahwa model belum menjelaskan variasi data dengan baik), nilai **R² pada NeuMF lebih tinggi**, menandakan performa yang relatif lebih stabil.

Secara keseluruhan, model **NeuMF** lebih direkomendasikan dalam konteks prediksi rating buku pada dataset ini karena menghasilkan prediksi yang lebih akurat.

## ✅ Kesimpulan dan Rencana Pengembangan ke Depan
---

Dalam proyek ini, dua pendekatan collaborative filtering — **RecommenderNet** dan **NeuMF** — telah dibangun dan diuji menggunakan **dataset buku dari Kaggle**. Hasil evaluasi menunjukkan bahwa **NeuMF** memberikan prediksi rating yang lebih baik dibandingkan RecommenderNet, dilihat dari nilai **RMSE dan MAE yang lebih rendah**, serta **R² yang lebih stabil**.

🔍 **Temuan Utama:**
- NeuMF dapat menangkap interaksi kompleks antara pengguna dan item dengan lebih baik berkat kombinasi matrix factorization dan neural network.
- RecommenderNet masih memiliki potensi, namun perlu perbaikan untuk menangani variasi data yang tinggi.

🚀 **Rencana Pengembangan Selanjutnya:**
1. Menambahkan metrik evaluasi berbasis ranking seperti **Hit@K**, **Precision@K**, atau **NDCG** untuk mengukur relevansi top-N rekomendasi.
2. Mengintegrasikan metode **content-based filtering** untuk menangani masalah cold start (pengguna/item baru).
3. Melatih model pada dataset yang lebih besar atau menggunakan fitur tambahan seperti genre, penulis, atau sinopsis buku untuk meningkatkan akurasi.
4. Menerapkan regularisasi atau teknik tuning hiperparameter agar model lebih general dan tidak overfitting.

Dengan pengembangan lebih lanjut, sistem rekomendasi ini diharapkan mampu memberikan rekomendasi buku yang lebih personal, akurat, dan bermanfaat bagi pengguna.
