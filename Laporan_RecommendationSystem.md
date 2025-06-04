# Laporan Proyek Machine Learning - Collaborative Filtering pada Dataset Buku

## 📌 Project Overview

Dalam era digital yang semakin berkembang, sistem rekomendasi memiliki peranan penting dalam membantu pengguna menavigasi informasi dan pilihan produk yang sangat besar. Salah satu aplikasi populer dari sistem rekomendasi adalah pada layanan e-commerce atau platform penyedia buku. Dengan semakin banyaknya pilihan buku dan meningkatnya kebutuhan personalisasi pengalaman pengguna, sistem rekomendasi yang efektif menjadi solusi penting untuk meningkatkan _engagement_ dan kepuasan pengguna.

Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis _collaborative filtering_ menggunakan dataset buku yang berisi informasi metadata seperti judul, penulis, tahun terbit, jumlah halaman, rating, dan lain-lain. Fokus utama proyek ini adalah memprediksi buku yang kemungkinan besar akan disukai oleh pengguna berdasarkan kesamaan perilaku atau preferensi pengguna lain.

Dataset ini bersifat publik dan mencakup berbagai atribut penting, di antaranya:

- **user_id**: ID pengguna dalam format `Uxxxx`.
- **book_id**: ID buku dalam format `Bxxxx`.
- **rating**: Nilai rating dari pengguna, skala 1-5.
- **author**: Nama penulis buku.
- **publisher**: Nama penerbit buku.
- **book_title**: Judul buku.

Teknik _collaborative filtering_ yang digunakan dalam proyek ini berbasis pada pendekatan **neural network embedding**, di mana fitur-fitur numerik dari pengguna dan item (dalam hal ini: tahun terbit dan jumlah halaman sebagai proxy untuk pengguna dan item) dipetakan ke dalam vektor laten untuk mempelajari pola interaksi.

> 📚 Penelitian terdahulu menunjukkan bahwa pendekatan matrix factorization dan deep learning dapat memberikan hasil yang baik dalam membangun sistem rekomendasi, di antaranya:
>
> - Koren, Bell, & Volinsky (2009) — ["Matrix Factorization Techniques for Recommender Systems"](https://ieeexplore.ieee.org/abstract/document/5197422)
> - Covington et al. (2016) — ["Deep Neural Networks for YouTube Recommendations"](https://dl.acm.org/doi/abs/10.1145/2959100.2959190)

Dalam proyek ini, dua model _collaborative filtering_ yang dikembangkan adalah:

1. **RecommenderNet** — Model sederhana berbasis embedding dan _dot product_ antara fitur pengguna dan item.
2. **NeuMF (Neural Matrix Factorization)** — Kombinasi antara pendekatan GMF (_Generalized Matrix Factorization_) dan MLP (_Multilayer Perceptron_) yang lebih kompleks.

Model-model tersebut dilatih menggunakan data interaksi buatan antara `user_id` dan `book_id` untuk mempelajari hubungan laten antar entitas, dan menghasilkan rekomendasi buku berdasarkan prediksi skor atau rating.

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
  1. **Dot product** antara embedding pengguna (`user_id`) dan item (`book_id`) untuk memprediksi rating — mendekati baseline matrix factorization.
  2. **Neural Collaborative Filtering (NeuMF)** — model yang menggabungkan _Generalized Matrix Factorization (GMF)_ dan _Multi-Layer Perceptron (MLP)_ untuk meningkatkan performa prediksi.
- Menggunakan **TensorFlow dan Keras** sebagai framework utama dalam membangun dan melatih model.

## 📊 Data Understanding
---

Dataset yang digunakan dalam proyek ini adalah **Books Dataset** yang tersedia secara publik melalui [Kaggle](https://www.kaggle.com/datasets/programmer3/personalized-book-ratings-dataset). Dataset ini mencakup informasi bibliografis dan metrik evaluasi terhadap berbagai judul buku, dan cocok digunakan dalam proyek sistem rekomendasi berbasis konten maupun kolaboratif. Dataset berjumlah 6 kolom, 1034 baris dan dataset bersih tidak ada missing values, dataset ini diambil dari platform Kaggle.

Dataset terdiri dari satu file utama:

- `dataset_buku.csv`: berisi informasi deskriptif mengenai ribuan buku, seperti user_id, book_id, penulis, rating, penerbit, dan judul buku.

### 🔢 Fitur-fitur dalam `dataset_buku.csv`:

- **user_id**: ID pengguna dalam format `Uxxxx`.
- **book_id**: ID buku dalam format `Bxxxx`.
- **rating**: Nilai rating dari pengguna, skala 1-5.
- **author**: Nama penulis buku.
- **publisher**: Nama penerbit buku.
- **book_title**: Judul buku.

### 🧭 Distribusi Data

- `user_id` terdiri dari ID pengguna yang tampaknya disimulasikan (misalnya `U0058`, `U0014`, dll), dengan total variasi pengguna yang menunjukkan adanya data interaksi eksplisit antara pengguna dan buku.
- `book_id` merupakan identifier unik untuk setiap buku, seperti `B0007`, `B0144`, dll, dan digunakan sebagai basis item dalam sistem rekomendasi.
- `rating` berkisar dari **1** hingga **5**, menunjukkan preferensi pengguna terhadap suatu buku. Distribusi rating cenderung **condong ke arah nilai tinggi**, konsisten dengan tren umum dalam sistem rating lainnya.
- `author` dan `publisher` memberikan informasi tambahan yang dapat dimanfaatkan dalam sistem rekomendasi berbasis **content-based filtering**.
- `book_title` memberikan konteks naratif yang bisa diekstraksi menjadi fitur teks (misalnya melalui TF-IDF) untuk pendekatan berbasis konten.

### 📌 Catatan

Dataset ini **sudah menyediakan `user_id` eksplisit**, sehingga sistem rekomendasi berbasis **collaborative filtering** dapat dibangun secara langsung tanpa perlu asumsi tambahan.  
Namun, pendekatan **hybrid** juga dapat diterapkan dengan mengombinasikan informasi eksplisit dari interaksi pengguna (`user_id`, `rating`) dan konten buku (`author`, `publisher`, `book_title`).

### 📊 Visualisasi Distribusi Data Rating

Grafik batang di bawah ini menunjukkan frekuensi masing-masing nilai rating buku dari skala 0 hingga 5. Terlihat bahwa rating **4.0** merupakan yang paling sering muncul, diikuti oleh rating **3.0** dan **5.0**. Hal ini mengindikasikan bahwa sebagian besar pembaca memberikan penilaian yang cukup positif terhadap buku-buku yang mereka baca.

![Distribusi Rating](https://raw.githubusercontent.com/silviaazahro/Machine-Learning-Terapan-Part-2/main/distribusi%20rating%20(2).png)

### 🧱 Visualisasi Histogram Data Rating

Histogram di bawah ini menggambarkan penyebaran nilai rating dalam bentuk batang berwarna biru muda. Pola distribusinya mirip dengan grafik distribusi sebelumnya, di mana rating tinggi (khususnya 4.0) mendominasi. Ini menunjukkan bahwa persepsi pembaca terhadap buku dalam dataset ini cenderung positif.

![Histogram Rating](https://raw.githubusercontent.com/silviaazahro/Machine-Learning-Terapan-Part-2/main/histogram%20rating%20(2).png)

### 📦 Visualisasi Boxplot Data Rating

Boxplot berikut menampilkan ringkasan statistik nilai rating buku, termasuk nilai minimum, maksimum, kuartil, dan median. Median rating berada di sekitar **3.8**, menunjukkan bahwa mayoritas rating berkisar pada nilai tinggi. Beberapa outlier dengan rating rendah juga terlihat, mencerminkan adanya buku yang dianggap kurang memuaskan oleh sebagian pembaca.

![Boxplot Rating](https://raw.githubusercontent.com/silviaazahro/Machine-Learning-Terapan-Part-2/main/boxplot%20rating%20(3).png)

## Data Preparation
---
Beberapa tahapan dalam persiapan data yang dilakukan meliputi:
1. **Encoding**  
   Mengonversi variabel *user_id* dan *book_id* menjadi indeks bilangan bulat agar dapat dimanfaatkan dalam layer embedding. Layer embedding membutuhkan input berupa indeks numerik yang nantinya dipetakan ke dalam vektor representasi laten.
2. **Normalisasi**  
   Melakukan normalisasi pada data *rating* ke dalam rentang 0 hingga 1 karena model menggunakan fungsi aktivasi sigmoid di output layer. Hal ini penting agar output model sesuai dengan skala target yang diinginkan.
3. **Pemilihan Fitur**  
   Variabel input (*x*) dibentuk dari fitur *user_index* dan *book_index*, sedangkan variabel target (*y*) berasal dari *rating_norm* yang telah dinormalisasi.
4. **Pengacakan Data**  
   Data diacak terlebih dahulu menggunakan `data.sample(frac=1, random_state=42)` untuk memastikan bahwa distribusi data pada set pelatihan dan validasi bersifat acak dan tidak berpola, yang membantu meningkatkan generalisasi model.
5. **Pembagian Dataset**  
   Data dipisah menjadi set pelatihan (*training set*) dan set validasi (*validation set*) untuk mengukur performa model secara objektif serta menghindari overfitting.

Tahapan ini dilakukan agar data siap digunakan secara efektif dalam model deep learning.

## Modeling
---

Dalam proyek ini, dikembangkan dua jenis model rekomendasi berbasis collaborative filtering. Kedua pendekatan tersebut digunakan untuk membandingkan kinerja model klasik yang memakai matrix factorization dengan model deep learning yang menggunakan neural network, dengan tujuan mengevaluasi performa serta fleksibilitas masing-masing dalam konteks rekomendasi buku.

### 1. **Matrix Factorization dengan Embedding (Baseline)**

Model ini menerapkan pendekatan collaborative filtering klasik yang diimplementasikan dengan menggunakan layer embedding pada TensorFlow.

**Arsitektur:**
- Terdapat dua embedding layer, masing-masing untuk *user* dan *book*.  
- Skor kecocokan dihitung melalui operasi **dot product** antar embedding tersebut.  
- Ditambahkan bias dan fungsi aktivasi sigmoid agar output berada pada rentang 0 hingga 1.

     ```python
     class RecommenderNet(tf.keras.Model):
    def __init__(self, num_user, num_book, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_user = num_user
        self.num_book = num_book
        self.embedding_size = embedding_size

        # Embedding dan bias untuk user
        self.user_embedding = layers.Embedding(
            num_user,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_user, 1)

        # Embedding dan bias untuk book
        self.book_embedding = layers.Embedding(
            num_book,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.book_bias = layers.Embedding(num_book, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])

        # Dot product antara user dan book embeddings
        dot_user_book = tf.reduce_sum(user_vector * book_vector, axis=1, keepdims=True)

        # Menambahkan bias
        x = dot_user_book + user_bias + book_bias

        # Aktivasi sigmoid untuk output rating (0 - 1)
        return tf.nn.sigmoid(x)
     ```

**Keunggulan:**  
- Struktur model yang sederhana dan efisien.  
- Proses pelatihan berlangsung dengan cepat.  
- Ideal digunakan sebagai baseline atau model awal.

**Keterbatasan:**  
- Hubungan antara *user* dan *book* hanya bersifat linear.  
- Kurang mampu menangkap pola kompleks atau non-linear dalam preferensi pengguna.

### 2. **Neural Matrix Factorization (NeuMF)**

NeuMF adalah pendekatan yang mengkombinasikan dua jalur, yaitu Generalized Matrix Factorization (GMF) dan Multi-Layer Perceptron (MLP). Metode ini lebih fleksibel karena mampu menangkap hubungan non-linear antar embedding.

**Arsitektur:**
- Embedding untuk user dan book diproses melalui dua jalur berbeda:
  - **GMF**: menggunakan operasi dot product seperti pada model klasik.  
  - **MLP**: menggabungkan (concatenate) embedding kemudian melewati beberapa fully connected layer.  
- Output dari kedua jalur tersebut digabungkan dan diteruskan ke dense layer terakhir.  
- Fungsi aktivasi sigmoid digunakan pada output untuk menghasilkan skor prediksi.

     ```python
  from tensorflow.keras import Input, Model, layers
import tensorflow as tf

def get_NeuMF_model(num_users, num_books, mf_dim=8, mlp_layers=[64, 32, 16, 8], dropout=0.0):
    # Input layer
    user_input = Input(shape=(1,), name="user_input")
    book_input = Input(shape=(1,), name="book_input")

    # MF part embedding
    mf_user_embedding = layers.Embedding(num_users, mf_dim, name="mf_user_embedding")(user_input)
    mf_book_embedding = layers.Embedding(num_books, mf_dim, name="mf_book_embedding")(book_input)
    mf_user_embedding = layers.Flatten()(mf_user_embedding)
    mf_book_embedding = layers.Flatten()(mf_book_embedding)
    mf_vector = layers.multiply([mf_user_embedding, mf_book_embedding])

    # MLP part embedding
    mlp_embedding_dim = mlp_layers[0] // 2
    mlp_user_embedding = layers.Embedding(num_users, mlp_embedding_dim, name="mlp_user_embedding")(user_input)
    mlp_book_embedding = layers.Embedding(num_books, mlp_embedding_dim, name="mlp_book_embedding")(book_input)
    mlp_user_embedding = layers.Flatten()(mlp_user_embedding)
    mlp_book_embedding = layers.Flatten()(mlp_book_embedding)
    mlp_vector = layers.concatenate([mlp_user_embedding, mlp_book_embedding])

    # MLP hidden layers
    for idx, units in enumerate(mlp_layers[1:]):
        mlp_vector = layers.Dense(units, activation='relu', name=f"mlp_dense_{idx}")(mlp_vector)
        if dropout > 0:
            mlp_vector = layers.Dropout(dropout)(mlp_vector)

    # Gabungkan MF dan MLP
    neumf_vector = layers.concatenate([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = layers.Dense(1, activation="sigmoid", name="prediction")(neumf_vector)

    model = Model(inputs=[user_input, book_input], outputs=prediction)
    return model
     ```

**Keunggulan:**  
- Mampu menangkap pola interaksi yang lebih rumit dan kompleks.  
- Lebih fleksibel dengan memanfaatkan kemampuan neural network untuk belajar hubungan non-linear.

**Keterbatasan:**  
- Struktur model lebih kompleks dan proses pelatihan lebih lambat dibandingkan model klasik.  
- Memerlukan penyesuaian hyperparameter yang lebih teliti dan cermat.

### Rekomendasi Top-N dari Dataset Buku

Dari hasil perbandingan kedua model rekomendasi buku, terlihat bahwa keduanya sama-sama merekomendasikan beberapa judul unik dan relevan, seperti *Interesting take can* karya **Rebecca Harrington** dan *Front accept after* karya **Lee Singh**, yang muncul pada kedua model. Ini menunjukkan adanya konsistensi dalam mengenali preferensi pengguna terhadap buku-buku tertentu.

Namun, **NeuMF** cenderung menghasilkan skor prediksi yang lebih terkonsentrasi dan sedikit lebih rendah dibandingkan dengan **RecommenderNet**, mengindikasikan bahwa NeuMF mungkin lebih konservatif dalam menilai kecocokan antara pengguna dan buku, serta lebih eksploratif terhadap buku dengan tema-tema yang kurang umum seperti *Name sign day significant* dan *Artist feel perform full*.

Sementara itu, **RecommenderNet** menunjukkan kecenderungan merekomendasikan buku-buku dengan nilai prediksi yang lebih tinggi, seperti *Example trade increase attention though* oleh **Joel Morris** dan *Indeed sing* oleh **Wesley Cobb**, yang bisa jadi lebih bersifat mainstream dan mudah dikenali oleh model karena pola interaksi pengguna yang lebih eksplisit.

Meskipun **NeuMF** menghasilkan rekomendasi yang lebih beragam dan “unik” dalam hal isi dan genre, hasil prediksi menunjukkan bahwa **RecommenderNet** memiliki skor prediksi rating yang sedikit lebih tinggi, yang dapat mengindikasikan kecocokan model ini dalam konteks preferensi pengguna secara umum. Hal ini sejalan dengan karakteristik masing-masing arsitektur, di mana **RecommenderNet** (matrix factorization) lebih fokus pada hubungan laten pengguna-buku, sementara **NeuMF** mengombinasikan pendekatan eksplisit dan implisit untuk menangkap preferensi yang lebih kompleks.

Oleh karena itu, dalam konteks sistem rekomendasi berbasis prediksi rating buku yang mengutamakan generalisasi dan ketepatan prediksi, model **RecommenderNet** tetap menjadi pilihan yang lebih direkomendasikan untuk kasus ini.

| Rank | **RecommenderNet**                     | Predicted Score | Predicted Rating | **NeuMF**                        | Predicted Score | Predicted Rating |
|-------|--------------------------------------|-----------------|------------------|---------------------------------|-----------------|------------------|
| 1     | Example trade increase attention though | 0.5291          | 3.1162           | Three become note law            | 0.5021          | 3.0084           |
| 2     | Indeed sing                          | 0.5217          | 3.0868           | Seven material third owner chair | 0.5017          | 3.0069           |
| 3     | Interesting take can                | 0.5191          | 3.0762           | Interesting take can             | 0.5016          | 3.0065           |
| 4     | Doctor much bag civil               | 0.5176          | 3.0703           | Artist feel perform full        | 0.5014          | 3.0054           |
| 5     | Front accept after                  | 0.5171          | 3.0682           | Name sign day significant       | 0.5011          | 3.0044           |
| 6     | Herself manage                     | 0.5149          | 3.0597           | Along and lay                   | 0.5011          | 3.0043           |
| 7     | Own trade possible                  | 0.5144          | 3.0578           | Should rule                    | 0.5011          | 3.0043           |
| 8     | Early gun ask                     | 0.5144          | 3.0577           | Name sign day significant       | 0.5011          | 3.0042           |
| 9     | Model far do better               | 0.5140          | 3.0560           | Front accept after              | 0.5010          | 3.0038           |
| 10    | Of social democratic              | 0.5140          | 3.0560           | Car decision lot                | 0.5009          | 3.0035           |


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
| RMSE    | 1.415598       | 1.417750 |
| MAE     | 1.214512       | 1.209832 |
| R²      | 0.002117       | -0.000920|

**Analisis:**
- Model **RecommenderNet** dan **NeuMF** menunjukkan performa yang sangat mirip berdasarkan metrik RMSE dan MAE.
- NeuMF memiliki nilai **MAE** yang sedikit lebih rendah dibanding RecommenderNet, menandakan prediksi absolutnya sedikit lebih akurat secara rata-rata.
- Sebaliknya, RecommenderNet memiliki nilai **RMSE** sedikit lebih rendah, mengindikasikan kesalahan kuadrat rata-rata yang sedikit lebih kecil.
- Pada metrik **R²**, RecommenderNet memiliki nilai positif sangat kecil, sementara NeuMF sedikit negatif. Hal ini menunjukkan kedua model belum mampu menjelaskan variasi data dengan baik, namun RecommenderNet menunjukkan kemampuan yang sedikit lebih baik dalam menjelaskan variasi tersebut.
- Secara keseluruhan, perbedaan performa kedua model sangat kecil dan keduanya dapat dianggap hampir setara dalam konteks dataset ini.

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
