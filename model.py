#!/usr/bin/env python
# coding: utf-8

# ### IMPORT LIBRARY

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


# ### IMPORT DATASET

# In[28]:


seed = 42


# In[29]:


film = pd.read_csv('dataset_film.csv')
film.sample(frac=1, random_state=seed)
film.head()


# # DATA UNDERSTANDING

# In[30]:


film.info()


# Penjelasan : 
# Pada dataset dataset_film.csv diketahui memiliki 45.130 baris dan 26 kolom untuk datasetnya. Tipe data dalam dataset tersebut bisa diketahui terdapat float64, int64 dan object yang menunjukan bahwa Tipe data pada dataset tersebut merupakan numerikan dan kategorikal.

# In[31]:


film.shape


# In[32]:


genre_cols = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 
              'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 
              'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 
              'Thriller', 'War', 'Western']

genre_counts = film[genre_cols].sum()



plt.figure(figsize=(10, 7))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=1000)
plt.title('Graph of Movies by Genre')
plt.axis('equal')
plt.show()


# Penjelasan : Menampilkan jumlah film per genre untuk dapat mengetahui total dari jumlah film untuk setiap genre pada dataset film tersebut

# In[33]:


numeric_columns = film.select_dtypes(include=['float64', 'int64']).columns

# Mengambil kolom numerik dari DataFrame film
numeric_columns = film.select_dtypes(include=['float64', 'int64']).columns

# Dictionary untuk menyimpan nilai minimal dan maksimal
min_max_values = {}

# Mengisi dictionary dengan nilai minimal dan maksimal dari setiap kolom
for column in numeric_columns:
    min_value = film[column].min()
    max_value = film[column].max()
    min_max_values[column] = {'min': min_value, 'max': max_value}

# Mengonversi dictionary menjadi DataFrame untuk tampilan tabel
min_max_df = pd.DataFrame(min_max_values).T
min_max_df.columns = ['Nilai Minimal', 'Nilai Maksimal']  # Mengganti nama kolom

min_max_df.head()


# Penjelasan : 
# Pada data tersebut memberikan informasi tentang  gambaran tentang karakteristik film dalam dataset, termasuk durasi, rating, popularitas, tahun rilis, dan genre.

# ### Menampilkan Korelasi Pearson pada dataset Film

# In[34]:


pearson_corr = film.corr(method='pearson')
print("Pearson Correlation:")
print(pearson_corr)


# Penjelasan : 
# Pada tabel tersebut terdapat 2 hal penting, yaitu korelasi positif dan korelasi negatif, pada kode tersebut menbenadingkan antara korelasi runtime dan genre.
# 
# (+) Korelasi Positif memiliki ciri nilai > 0
# Contoh Korelasi : 
# - Pada Genre Action dan Adventure memiliki korelasi yang tinggi pada nilai (0.287) yang menunjukan bahwa film action sering dikategorikan sebagai film adventure
# 
# (-) Korelasi Negatif meiliki ciri nilai < 0
# Contoh Korelasi :
# - Pada Genre film Comedy dan Thriler memiliki korelasi yang negatif dengan nilai (-0,204) yang menunjukan bahwa kedua genre tersebut kurang berkaitan

# ### Menampilkan Korelasi dengan menggunakan Visualisasi Data Heatmap

# In[35]:


sns.heatmap(pearson_corr, annot=True, cmap='coolwarm')
plt.title("Pearson Correlation Heatmap")
plt.show()


# In[36]:


missing_values = film.isnull().sum()
print("Missing Values per Kolom:")
print(missing_values[missing_values > 0])

numeric_columns = film.select_dtypes(include=['float64', 'int64']).columns


# Penjelasan : 
# - Warna merah untuk korelasi positif mendekati nilai (+1)
# - Warna biru untuk korelasi negatif mendekati niali (-1)
# - Warna putih memberikan informasi tidak ada korelasi dengan nilai.
# 
# Hubungan Antar Fitur : 
# - Pada heatmap tersebut, dapat dikethai bahwa genre - genre dari film tersebut berkaitan satu sama lain, seperti Family dengan Animation yang memiliki korelasi positif.
# 
# - Pada genre Documentary memiliki korelasi negatif dengan genre Action, yang memberikan informasi bahwa kedua genre tersebut tidak memiliki korelasi dalam satu film bersamaan.

# ### Visualisasi dengan Boxplot & Mengidentifikasi Outliar

# In[37]:


num_columns = len(numeric_columns)
n_rows = (num_columns // 5) + (num_columns % 5 > 0)  # Calculate number of rows needed

plt.figure(figsize=(15, n_rows * 5))  # Adjust height based on the number of rows
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(n_rows, 5, i)  # Adjust the number of rows accordingly
    sns.boxplot(x=film[column])
    plt.title(f'Boxplot: {column}')

plt.tight_layout()
plt.show()

# Menghitung Z-Score untuk mengidentifikasi outlier
z_scores = np.abs(stats.zscore(film[numeric_columns]))

# Menentukan threshold untuk outlier
threshold = 3
outliers = (z_scores > threshold).sum(axis=0)

# Menampilkan jumlah outlier per kolom
outlier_counts = pd.Series(outliers, index=numeric_columns)
print("\nJumlah Outlier per Kolom:")
print(outlier_counts[outlier_counts > 0])  # Menampilkan kolom yang memiliki outlier


# Penjelasan : 
# Penggunaan Boxplot pada analisis dataset film tersebut bertujuan untuk dapat mengetahui dan mendeteksi outliar dalam data.
# 
# - Pada variabel (Numerik)  runtime, vote_count dan release_year diketahui pada boxplot tersebut memiliki banyak outliar, dengan menunjukan bahwa ada berberapa film yang durasi, jumlah vote dan tahun rilisnya jauh berbeda dari rata - rata yang diinginkan
# 

# In[38]:


z_scores = np.abs(stats.zscore(film[numeric_columns]))

# Menentukan threshold untuk outlier
threshold = 3
outliers = (z_scores > threshold).sum(axis=0)

# Jumlah Outlier tiap Kolom
outlier_counts = pd.Series(outliers, index=numeric_columns)
print("\nJumlah Outlier per Kolom:")
print(outlier_counts[outlier_counts > 0])


# ### Data Cleaning

# In[39]:


# Visualisasi pola distribusi film 
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_film['runtime'], bins=30, kde=True)
plt.title('Distribusi Durasi Film')
plt.xlabel('Durasi (Menit)')
plt.ylabel('Jumlah Film')
plt.grid(axis='y')
plt.axvline(cleaned_film['runtime'].mean(), color='r', linestyle='--', label='Rata-rata')
plt.axvline(cleaned_film['runtime'].median(), color='g', linestyle='--', label='Median')
plt.legend()
plt.show()


# Penjelasan : 
# - Melihat pola distribusi durasi film dalam dataset.
# - Melihat nilai rata-rata dan median durasi film.

# In[40]:


mask = (z_scores < threshold).all(axis=1)
cleaned_film = film[mask]

# Menampilkan jumlah data setelah pembersihan
print(f"\nJumlah data sebelum pembersihan: {len(film)}")
print(f"Jumlah data setelah pembersihan: {len(cleaned_film)}")


# In[41]:


# Menghitung z-score pada data yang dibersihkan
z_scores_cleaned = np.abs(stats.zscore(cleaned_film[numeric_columns]))

# Menentukan jumlah outlier baru
outliers_cleaned = (z_scores_cleaned > threshold).sum(axis=0)
outlier_counts_cleaned = pd.Series(outliers_cleaned, index=numeric_columns)

print("\nJumlah Outlier per Kolom Setelah Pembersihan:")
print(outlier_counts_cleaned[outlier_counts_cleaned > 0])  


# # DATA PREPROCESSING & PREPARATION

# In[42]:


z_scores = np.abs(stats.zscore(film[numeric_columns]))
threshold = 3
mask = (z_scores < threshold).all(axis=1)
cleaned_film = film[mask]

# Binning untuk kolom runtime dengan label baru
bins = [0, 60, 120, 180, np.inf]  # Durasi dalam menit
labels = ['Short', 'Medium', 'Long', 'Very Long']
cleaned_film['runtime_binned'] = pd.cut(cleaned_film['runtime'], bins=bins, labels=labels, right=False)

# Menampilkan distribusi bin
plt.figure(figsize=(8, 5))
sns.countplot(data=cleaned_film, x='runtime_binned', order=labels)
plt.title('Distribusi Durasi Film setelah Binning')
plt.xlabel('Durasi Film (Binned)')
plt.ylabel('Jumlah Film')
plt.show()


# Penjelasan : 
# Mengidentikasi Binning pada runtime dengan durasi film yang dibagi menjadi 4 tipe bins : 
# - 0 hingga 60 menit (Short)
# - 60 hingga 120 menit (Medium)
# - 120 hingga 180 menit (Long)
# - Lebih dari 180 menit (Very Long)

# In[43]:


numeric_columns = cleaned_film.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()

# Melakukan normalisasi
cleaned_film[numeric_columns] = scaler.fit_transform(cleaned_film[numeric_columns])

# Menampilkan beberapa baris data setelah normalisasi
cleaned_film.head()


# Penjelasan : 
# 
# Pada dataset tersebut tabel yang akan dilakukan normalisasi adalah kolom runtime, vote_average dan vote_count dan kolom numerik lainya yang tersisa setelah mengapus outliar.
# 
# Dengan normalisasi data tersebut DataFrame dapat dilakukan ke tahap modeling untuk dianalisis lebih lanjut

# # Modeling

# In[44]:


X = cleaned_film.drop(['title', 'genres', 'runtime_binned'], axis=1)  
y = cleaned_film['runtime_binned']  


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[46]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)


# In[47]:


y_pred = model.predict(X_test)


# # Evaluate

# In[48]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[49]:


import pickle
import joblib


# In[50]:


with open('modelKNN1.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[51]:


import joblib

joblib.dump(model, "knn_model.sav")


# In[ ]:




