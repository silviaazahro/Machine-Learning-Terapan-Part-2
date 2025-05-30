# -*- coding: utf-8 -*-
"""RecommendationSystem.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NnhI6ECKhj-gTO3s_3mLZL5ERIzmrTMt

# **1. Import Library**

Mengimport library yang dibutuhkan
"""

!"{sys.executable}" -m pip install matplotlib-venn

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
from sklearn.manifold import TSNE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from wordcloud import WordCloud

"""# **2. Load Dataset**

Membaca dataset buku.
"""

data = pd.read_csv('https://raw.githubusercontent.com/silviaazahro/Machine-Learning-Terapan-Part-2/refs/heads/main/dataset/dataset_buku.csv')

"""Menampilkan sample data"""

data.head()

"""# **3. Data Wrangling**

Mengecek apakah ada missing value
"""

data.isnull().sum()

"""Masih terdapat Missing Value, maka kita perlu menghilangkan Missing Value pada kolom **subtitle, authors, categories, thumbnail, description, description, published_year, average_rating, num_pages, ratings_count** dengan menggunakan imputas Mean dan Median."""

data['subtitle'] = data['subtitle'].fillna('')
data['authors'] = data['authors'].fillna('Unknown')
data['categories'] = data['categories'].fillna('Other')
data['thumbnail'] = data['thumbnail'].fillna('https://example.com/default-thumbnail.jpg')
data['description'] = data['description'].fillna('No description available')
data['published_year'] = data['published_year'].fillna(data['published_year'].median())
data['average_rating'] = data['average_rating'].fillna(data['average_rating'].mean())
data['num_pages'] = data['num_pages'].fillna(data['num_pages'].median())
data['ratings_count'] = data['ratings_count'].fillna(0)

"""Mengecek kembali apakah masih terdapat Missing Value atau tidak"""

data.isnull().sum()

"""# **4. EDA**

Melihat distribusi data pada fitur rating
"""

plt.figure(figsize=(10, 6))
sns.countplot(x='average_rating', data=data, palette='viridis')
plt.title('Distribusi Rating')
plt.xlabel('Rating')
plt.ylabel('Frekuensi')
plt.show()

"""Melihat histogram dari fitur rating"""

plt.figure(figsize=(10, 6))
plt.hist(data['average_rating'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram Rating')
plt.xlabel('Rating')
plt.ylabel('Frekuensi')
plt.show()

"""Memvisualisasi fitur rating dengan boxplot untuk melihat korelasi atau insight lain."""

plt.figure(figsize=(10,6))
sns.boxplot(x='average_rating', data=data, palette='viridis')
plt.title('Boxplot Rating')
plt.xlabel('Rating')
plt.show()

"""# **4. Data Preparation**

Mengambil daftar unik tahun publish dan jumlah halaman
"""

# Ambil nilai unik dari dataset
unique_year = data['published_year'].unique().tolist()
unique_pages = data['num_pages'].unique().tolist()

"""Membuat dictionary untuk mapping published_year dan num_pages ke integer"""

# Buat mapping dari nilai asli ke indeks
year_to_index = {year: idx for idx, year in enumerate(sorted(unique_year))}
pages_to_index = {pages: idx for idx, pages in enumerate(sorted(unique_pages))}

# Buat mapping dari indeks ke nilai asli
index_to_year = {idx: year for year, idx in year_to_index.items()}
index_to_pages = {idx: pages for pages, idx in pages_to_index.items()}

"""Mapping ke dataframe"""

# Mapping kolom ke indeks di DataFrame
data['year_index'] = data['published_year'].map(year_to_index)
data['pages_index'] = data['num_pages'].map(pages_to_index)

"""Mengambil nilai min dan max dari rating"""

min_rating = data['average_rating'].min()
max_rating = data['average_rating'].max()

"""Normalisasi rating ke skala 0-1"""

data['rating_norm'] = data['average_rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))

"""Mendeklarasikan kolom fitur dan target"""

# Input: kolom 'published_year' dan 'num_pages'
x = data[['year_index', 'pages_index']].values

# Target: rating yang sudah dinormalisasi
y = data['rating_norm'].values

"""# **5. Data Splitting**

Mengacak data
"""

data = data.sample(frac=1, random_state=42)
x = data[['year_index', 'pages_index']].values
y = data['rating_norm'].values

"""Membagi ke training dan validasi

"""

train_size = int(0.8 * len(data))
x_train, x_val = x[:train_size], x[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

print("Jumlah data training:", x_train.shape[0])
print("Jumlah data validasi:", x_val.shape[0])

"""# **6. Model Building**

## 6.1. RecommenderNet

Melakukan model building dengan RecommenderNet dan Mengkompilasi model dengan Binary Crossentropy dan menggunakan optimizer Adam serta metric RMSE.
"""

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

num_year = len(year_to_index)
num_pages = len(pages_to_index)
embedding_size = 20

model = RecommenderNet(
    num_year,
    num_pages,
    embedding_size
)

model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""## 6.2. NeuMF"""

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

# Inisialisasi NeuMF:
num_years = len(year_to_index)
num_pages = len(pages_to_index)

neuMF_model = get_NeuMF_model(
    num_years, num_pages,
    mf_dim=8,
    mlp_layers=[64,32,16,8]
)

neuMF_model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""# **7. Model Training**

Menerapkan Callbacks early stoping untuk menghentikan model ketika val_loss tidak berkurang selama 3 epoch dan merestore kembali weight yang terbaik.
"""

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

"""## 7.1. RecommenderNet

Melatih model RecommenderNet.
"""

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

"""## 7.2. NeuMF

Melatih model NeuMF.
"""

history_neuMF = neuMF_model.fit(
    [x_train[:,0], x_train[:,1]], y_train,
    batch_size=32,
    epochs=20,
    validation_data=([x_val[:,0], x_val[:,1]], y_val),
    callbacks=callbacks
    )

"""# **8. Model Evaluation**

## 8.1. Plot History Training

Membandingkan dua model rekomendasi, RecommenderNet dan NeuMF, dengan menampilkan grafik yang menunjukkan bagaimana loss dan RMSE berubah selama pelatihan.
"""

# Membuat figure dan axes untuk dua plot berdampingan
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot pertama (RecommenderNet)
ax1.plot(history.history['loss'], label='Loss Train')
ax1.plot(history.history['val_loss'], label='Loss Validasi')
ax1.plot(history.history['root_mean_squared_error'], label='RMSE Train')
ax1.plot(history.history['val_root_mean_squared_error'], label='RMSE Validasi')
ax1.set_title('Grafik Loss dan RMSE RecommenderNet')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Nilai')
ax1.legend()
ax1.grid(True)

# Plot kedua (NeuMF)
ax2.plot(history_neuMF.history['loss'], label='Loss Train')
ax2.plot(history_neuMF.history['val_loss'], label='Loss Validasi')
ax2.plot(history_neuMF.history['root_mean_squared_error'], label='RMSE Train')
ax2.plot(history_neuMF.history['val_root_mean_squared_error'], label='RMSE Validasi')
ax2.set_title('Grafik Loss dan RMSE NeuMF')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Nilai')
ax2.legend()
ax2.grid(True)

# Menampilkan plot
plt.tight_layout()
plt.show()

"""## 8.2. Metrik Error (RMSE, MAE, R²)

Membandingkan kinerja dua model rekomendasi, RecommenderNet dan NeuMF, dengan menghitung beberapa metrik evaluasi seperti RMSE, MAE, dan R².
"""

# Prediksi dan rescaling untuk model RecommenderNet
y_pred = model.predict(x_val).flatten()
y_pred_rescaled = y_pred * (max_rating - min_rating) + min_rating
y_val_rescaled = y_val * (max_rating - min_rating) + min_rating

mse_recommender = mean_squared_error(y_val_rescaled, y_pred_rescaled)
rmse_recommender = np.sqrt(mse_recommender)
mae_recommender = mean_absolute_error(y_val_rescaled, y_pred_rescaled)
r2_recommender = r2_score(y_val_rescaled, y_pred_rescaled)

# Prediksi dan rescaling untuk model NeuMF
year_input = x_val[:, 0]
pages_input = x_val[:, 1]
y_pred_neuMF = neuMF_model.predict([year_input, pages_input]).flatten()

y_pred_rescaled_neuMF = y_pred_neuMF * (max_rating - min_rating) + min_rating
y_val_rescaled_neuMF = y_val * (max_rating - min_rating) + min_rating

mse_neuMF = mean_squared_error(y_val_rescaled_neuMF, y_pred_rescaled_neuMF)
rmse_neuMF = np.sqrt(mse_neuMF)
mae_neuMF = mean_absolute_error(y_val_rescaled_neuMF, y_pred_rescaled_neuMF)
r2_neuMF = r2_score(y_val_rescaled_neuMF, y_pred_rescaled_neuMF)

# Membuat DataFrame untuk perbandingan
comparison_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R²"],
    "RecommenderNet": [rmse_recommender, mae_recommender, r2_recommender],
    "NeuMF": [rmse_neuMF, mae_neuMF, r2_neuMF]
})

comparison_df

"""## 8.3. Scatterplot : Actual Rating vs Predicted Rating

Membandingkan hasil prediksi rating dari dua model rekomendasi, RecommenderNet dan NeuMF, dengan menggunakan scatter plot. Garis merah pada kedua plot menunjukkan garis referensi di mana prediksi dan rating sebenarnya akan saling bertemu. Hal ini dapat membantu melihat seberapa akurat prediksi masing-masing model dibandingkan dengan rating yang sebenarnya.
"""

# Membuat figure dan axes untuk dua scatter plot berdampingan
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot pertama (RecommenderNet)
ax1.scatter(y_val_rescaled, y_pred_rescaled, alpha=0.3, color='dodgerblue')
ax1.plot([min_rating, max_rating], [min_rating, max_rating], color='red', linestyle='--')
ax1.set_xlabel('Rating Sebenarnya')
ax1.set_ylabel('Rating Prediksi')
ax1.set_title('Scatter Plot: Rating Sebenarnya vs Prediksi RecommenderNet')
ax1.grid(True)

# Plot kedua (NeuMF)
ax2.scatter(y_val_rescaled_neuMF, y_pred_rescaled_neuMF, alpha=0.3, color='dodgerblue')
ax2.plot([min_rating, max_rating], [min_rating, max_rating], color='red', linestyle='--')
ax2.set_xlabel('Rating Sebenarnya')
ax2.set_ylabel('Rating Prediksi')
ax2.set_title('Scatter Plot: Rating Sebenarnya vs Prediksi NeuMF')
ax2.grid(True)

# Menampilkan plot
plt.tight_layout()
plt.show()

"""## 8.4. Top-N

Membandingkan rekomendasi buku dari dua model, RecommenderNet dan NeuMF, berdasarkan prediksi yang dihasilkan oleh masing-masing model.
"""

# Fungsi untuk mendapatkan rekomendasi Top-N
def get_top_n_recommendations(year_id_asli, model, n=10, use_neumf=False):
    if year_id_asli not in year_to_index:
        raise ValueError(f"Tahun {year_id_asli} tidak ditemukan dalam data. Coba salah satu dari: {list(year_to_index.keys())[:5]}...")

    year_index = year_to_index[year_id_asli]
    all_pages_indices = np.array(list(index_to_pages.keys()))

    books_in_year_pages = data[data['published_year'] == year_id_asli]['num_pages'].tolist()
    readed_indices = [pages_to_index[page] for page in books_in_year_pages if page in pages_to_index]
    unreaded_indices = list(set(all_pages_indices) - set(readed_indices))

    # Membuat input untuk model
    year_input = np.full(len(unreaded_indices), year_index)
    pages_input_for_prediction = np.array(unreaded_indices)
    if use_neumf:
        predictions = model.predict([year_input, pages_input_for_prediction], verbose=0).flatten()
    else:
        input_pairs = np.stack((year_input, pages_input_for_prediction), axis=1)
        predictions = model.predict(input_pairs, verbose=0).flatten()


    top_prediction_indices_in_unreaded = np.argsort(predictions)[-n:][::-1]
    top_books_indices_from_unreaded = pages_input_for_prediction[top_prediction_indices_in_unreaded]
    top_books_pages = [index_to_pages[idx] for idx in top_books_indices_from_unreaded]
    top_predictions = predictions[top_prediction_indices_in_unreaded]

    recommended_data = []
    for i, page_index in enumerate(top_books_indices_from_unreaded):
        page_value = index_to_pages[page_index]
        book_info = data[data['num_pages'] == page_value].iloc[0]
        recommended_data.append({
            "pages_index": page_index,
            "Predicted Score": top_predictions[i],
            "title": book_info['title'],
            "published_year": book_info['published_year'],
            "num_pages": book_info['num_pages']
        })

    top_books_df = pd.DataFrame(recommended_data)

    # Rescale the predicted score to the original rating range
    top_books_df["Predicted Rating"] = top_books_df["Predicted Score"] * (max_rating - min_rating) + min_rating

    # Sort and add rank
    top_books_df = top_books_df.sort_values(by="Predicted Score", ascending=False).reset_index(drop=True)
    top_books_df["Rank"] = range(1, len(top_books_df) + 1)

    # Select the final columns
    return top_books_df[["Rank", "title", "Predicted Score", "Predicted Rating", "published_year", "num_pages"]]

# Pemanggilan untuk mengambil rekomendasi top-10 dari model pertama
top_n_rec_recommender = get_top_n_recommendations(2010, model, n=10, use_neumf=False)

# Pemanggilan untuk mengambil rekomendasi top-10 dari model kedua (NeuMF)
top_n_rec_neumf = get_top_n_recommendations(2010, neuMF_model, n=10, use_neumf=True)

# Gabungkan hasil dari kedua model
top_n_rec_recommender['Model'] = 'RecommenderNet'
top_n_rec_neumf['Model'] = 'NeuMF'

# Gabungkan hasil kedua DataFrame tanpa menduplikasi data
top_n_combined = pd.concat([top_n_rec_recommender, top_n_rec_neumf], ignore_index=True)

# Menampilkan hasil
top_n_combined

"""Membandingkan rekomendasi buku dari dua model, RecommenderNet dan NeuMF, dalam bentuk grafik horizontal yang menampilkan skor prediksi dari setiap buku yang direkomendasikan oleh kedua model."""

import matplotlib.pyplot as plt

# Ambil skor dari hasil DataFrame untuk RecommenderNet
top_scores_recommender = top_n_rec_recommender['Predicted Rating'].values
top_titles_recommender = top_n_rec_recommender['title'].values

# Ambil skor dari hasil DataFrame untuk NeuMF
top_scores_neumf = top_n_rec_neumf['Predicted Rating'].values
top_titles_neumf = top_n_rec_neumf['title'].values

# Membuat sub-plot untuk perbandingan
plt.figure(figsize=(14, 6))

# Plot Top-10 rekomendasi RecommenderNet
plt.subplot(1, 2, 1)
plt.barh(top_titles_recommender, top_scores_recommender, color='skyblue')
plt.xlabel('Skor Prediksi')
plt.title('Top-10 Buku yang Direkomendasikan (RecommenderNet)')
plt.gca().invert_yaxis()

# Plot Top-10 rekomendasi NeuMF
plt.subplot(1, 2, 2)
plt.barh(top_titles_neumf, top_scores_neumf, color='lightgreen')
plt.xlabel('Skor Prediksi')
plt.title('Top-10 Buku yang Direkomendasikan (NeuMF)')
plt.gca().invert_yaxis()

# Menampilkan kedua plot
plt.tight_layout()
plt.show()

"""## 8.5. Visualisasi 2D Embedding

Membandingkan embedding buku yang dihasilkan oleh dua model rekomendasi, RecommenderNet dan NeuMF, dengan memvisualisasikan hasil reduksi dimensi menggunakan t-SNE.
"""

# Mengambil matriks embedding dari model pertama (RecommenderNet)
books_embeddings_recommender = model.pages_embedding.get_weights()[0]  # bentuk: (num_pages, embedding_size)

# Reduksi dimensi ke 2D dengan t-SNE untuk model pertama
tsne_recommender = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
books_embeddings_2d_recommender = tsne_recommender.fit_transform(books_embeddings_recommender)

# DataFrame untuk plot model pertama
df_emb_recommender = pd.DataFrame(books_embeddings_2d_recommender, columns=['x', 'y'])
df_emb_recommender['pages_index'] = np.arange(books_embeddings_recommender.shape[0])

# Mengambil matriks embedding dari model kedua (NeuMF)
books_embedding_layer_neumf = neuMF_model.get_layer("mf_page_embedding")
books_embeddings_neumf = books_embedding_layer_neumf.get_weights()[0]

# Reduksi dimensi ke 2D dengan t-SNE untuk model kedua
tsne_neumf = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
books_embeddings_2d_neumf = tsne_neumf.fit_transform(books_embeddings_neumf)

# DataFrame untuk plot model kedua
df_emb_neumf = pd.DataFrame(books_embeddings_2d_neumf, columns=['x', 'y'])
df_emb_neumf['pages_index'] = np.arange(books_embeddings_neumf.shape[0])

# Membaca data
data['pages_index'] = data['num_pages'].map(pages_to_index)

# Menggabungkan data embedding dengan pages info untuk kedua model
df_plot_recommender = pd.merge(df_emb_recommender, data, on='pages_index', how='left')
df_plot_neumf = pd.merge(df_emb_neumf, data, on='pages_index', how='left')

# Membuat figure untuk visualisasi gabungan
plt.figure(figsize=(16, 8))

# Plot untuk model pertama (RecommenderNet) di subplot kiri
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_plot_recommender, x='x', y='y', hue='categories', palette='tab10', legend=False, s=50)
plt.title('Embedding Buku (RecommenderNet)')
plt.xlabel('Dimensi 1')
plt.ylabel('Dimensi 2')

# Plot untuk model kedua (NeuMF) di subplot kanan
plt.subplot(1, 2, 2)
sns.scatterplot(data=df_plot_neumf, x='x', y='y', hue='categories', palette='Set2', legend=False, s=50)
plt.title('Embedding Buku (NeuMF)')
plt.xlabel('Dimensi 1')
plt.ylabel('Dimensi 2')

# Menampilkan plot
plt.tight_layout()
plt.show()

"""## 8.6. Venn Diagram

Membandingkan hasil rekomendasi dari dua model, RecommenderNet dan NeuMF, dengan menggunakan Diagram Venn.
"""

# Ambil Top-10 rekomendasi dari model pertama (RecommenderNet)
top_n_rec_recommender = get_top_n_recommendations(2012, model, n=10, use_neumf=False)
recommended_books_recommender = set(top_n_rec_recommender['title'])

# Ambil Top-10 rekomendasi dari model kedua (NeuMF)
top_n_rec_neumf = get_top_n_recommendations(2012, neuMF_model, n=10, use_neumf=True)
recommended_books_neumf = set(top_n_rec_neumf['title'])

# Membuat Diagram Venn untuk membandingkan rekomendasi kedua model
plt.figure(figsize=(8, 6))
venn2([recommended_books_recommender, recommended_books_neumf],
      set_labels=('RecommenderNet', 'NeuMF'))

plt.title('Perbandingan Rekomendasi Buku: RecommenderNet vs NeuMF')
plt.show()

"""## 8.7. WordCloud

Visualisasi WordCloud yang membandingkan judul buku yang direkomendasikan oleh dua model, RecommenderNet dan NeuMF.
"""

# Gabungkan judul buku yang direkomendasikan dari model pertama (RecommenderNet)
top_titles_recommender = top_n_rec_recommender['title'].values
text_recommender = " ".join(top_titles_recommender)

# Buat word cloud untuk model pertama (RecommenderNet)
wordcloud_recommender = WordCloud(width=800, height=400, background_color='white').generate(text_recommender)

# Gabungkan judul buku yang direkomendasikan dari model kedua (NeuMF)
top_titles_neumf = top_n_rec_neumf['title'].values
text_neumf = " ".join(top_titles_neumf)

# Buat word cloud untuk model kedua (NeuMF)
wordcloud_neumf = WordCloud(width=800, height=400, background_color='white').generate(text_neumf)

# Membuat figure untuk visualisasi gabungan
plt.figure(figsize=(16, 8))

# Plot WordCloud untuk model pertama (RecommenderNet) di sebelah kiri
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_recommender, interpolation='bilinear')
plt.axis('off')
plt.title('Top-10 Buku yang Direkomendasikan (RecommenderNet)')

# Plot WordCloud untuk model kedua (NeuMF) di sebelah kanan
plt.subplot(1, 2, 2)
plt.imshow(wordcloud_neumf, interpolation='bilinear')
plt.axis('off')
plt.title('Top-10 Buku yang Direkomendasikan (NeuMF)')

# Menampilkan plot
plt.show()