# main.py
# Analisis Sentimen Bahasa Indonesia - UTS NLP
# Mengacu pada PDF: NLP 1, NLP 2, NLP 3, NLP 4

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import matplotlib.pyplot as plt

# === 1. Baca dataset ===
df = pd.read_csv('tweets_comifuro21.csv')
print("Jumlah data:", len(df))
print("Distribusi label:\n", df['label'].value_counts())

# === 2. Pra-pemrosesan (NLP 2 & NLP 3) ===

stopwords_id = {
    # Kata umum yang tidak berpengaruh pada sentimen
    'yg', 'nya', 'di', 'ke', 'dari', 'ini', 'itu', 'dan', 'ada', 'untuk',
    'saya', 'kamu', 'kita', 'mereka', 'dengan', 'oleh', 'pada', 'kepada',
    'dalam', 'akan', 'bisa', 'tidak', 'ga', 'gak', 'enggak', 'banget',
    'bgt', 'sih', 'aja', 'doang', 'saja', 'juga', 'ya', 'loh', 'dong',
    'nih', 'deh', 'kalo', 'kalau', 'gue', 'lu', 'gw', 'loh', 'mah', 'aja',
    'aja', 'terus', 'lagi', 'udah', 'dah', 'emang', 'bener', 'tau', 'tahu',
    'gitu', 'kayak', 'terlalu', 'bikin', 'jadi', 'aja', 'masih', 'sudah',
    'lebih', 'bisa', 'harus', 'mesti', 'waktu', 'pas', 'kayanya', 'mungkin',
    'aja', 'banget', 'bangettt', 'sih', 'nih', 'dong', 'halo', 'hai'
}

# Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Hapus URL, mention, hashtag, angka, tanda baca
    text = re.sub(r'http\S+|@\w+|#\w+|\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenisasi & hapus stopword
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords_id]
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess)
df = df[df['clean_text'] != ""]

print("\nContoh hasil preprocessing:")
print(df[['text', 'clean_text']].head())

# === 3. Vektorisasi TF-IDF (NLP 3) ===
X = df['clean_text']
y = df['label']

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

# === 4. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# === 5. Model: Naive Bayes (NLP 1) ===
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("\n=== Naive Bayes ===")
print("Akurasi:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# === 6. Prediksi semua data untuk analisis akhir ===
y_all_pred = nb_model.predict(X_tfidf)
df['predicted'] = y_all_pred

# Hitung persentase
total = len(df)
pos = len(df[df['predicted'] == 'positif'])
neg = len(df[df['predicted'] == 'negatif'])

print(f"\n📊 ANALISIS SENTIMEN AKHIR:")
print(f"Total  {total}")
print(f"Positif: {pos} ({pos/total*100:.1f}%)")
print(f"Negatif: {neg} ({neg/total*100:.1f}%)")

# === 7. Simpan hasil ===
df.to_csv('hasil_sentimen.csv', index=False)
print("✅ File 'hasil_sentimen.csv' berhasil disimpan!")

# === 8. Visualisasi (opsional) ===
plt.figure(figsize=(6, 6))
plt.pie([pos, neg], labels=['Positif', 'Negatif'], colors=['green', 'red'], autopct='%1.1f%%')
plt.title('Distribusi Sentimen #Comifuro21')
plt.savefig('sentiment_pie.png')
plt.show()