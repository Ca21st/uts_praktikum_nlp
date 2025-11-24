🎌 Analisis Sentimen #Comifuro21 – UTS Natural Language Processing
Nama: Abdillah Fakhri Santosa
NIM: 122005011
Mata Kuliah: Natural Language Processing (IF400105)
Dosen: Tori Sutisna, S.T., M.Kom.
Prodi: Teknik Informatika – Universitas Halim Sanusi PUI Bandung


📌 Deskripsi Proyek
Proyek ini bertujuan untuk melakukan analisis sentimen terhadap komentar publik di media sosial (Twitter/X) mengenai Comifuro21, sebuah event komunitas pop culture terbesar di Indonesia. Analisis dilakukan terhadap 103 komentar dalam Bahasa Indonesia, yang terdiri dari:

Data nyata yang dikumpulkan secara manual dari unggahan publik di X
Data sintetis yang merepresentasikan pola umum opini pengguna
Tujuan akhir: mengklasifikasikan sentimen sebagai positif atau negatif, lalu menghitung persentase distribusinya.


🧠 Metodologi (Sesuai PDF NLP 1–4)
Proses analisis mengikuti alur NLP standar:

Pengumpulan Data: 103 komentar publik bertopik #Comifuro21
Pembersihan Data:
Hapus URL, mention, hashtag, emoji, angka, dan tanda baca
Gunakan regex sesuai NLP 2 & NLP 3
Preprocessing:
Lowercasing
Tokenisasi
Stopword removal (custom untuk Bahasa Indonesia)
Stemming menggunakan Sastrawi
Vektorisasi: TF-IDF (TfidfVectorizer) dengan n-gram (1–2)
Pemodelan: Klasifikasi dengan Naive Bayes (MultinomialNB)
Evaluasi & Analisis:
Akurasi model
Persentase sentimen akhir (positif vs negatif)
Visualisasi pie chart


📊 Hasil
Total data: 103 komentar
Distribusi label asli:
Positif: 63
Negatif: 40
Akurasi model Naive Bayes: ±85–90% (bervariasi tergantung split)
Hasil prediksi akhir:
✅ Positif: ~62%
❌ Negatif: ~38%

▶️ Cara Menjalankan
Instal dependensi:
pip install pandas scikit-learn Sastrawi matplotlib
Jalankan program:
python main.py
Program akan:
Menampilkan jumlah data & distribusi label
Menampilkan akurasi dan laporan klasifikasi
Menyimpan hasil_sentimen.csv dan sentiment_pie.png
Menunggu pengguna menekan Enter sebelum keluar

📚 Referensi
PDF Materi NLP 1–4: Klasifikasi Teks, Preprocessing, TF-IDF, Naive Bayes
Sastrawi: https://github.com/sastrawi/sastrawi
Scikit-learn: https://scikit-learn.org
Pandas & Matplotlib: dokumentasi resmi

📎 Lampiran
Dataset: Kombinasi data nyata dan sintetis (semua dalam Bahasa Indonesia)
Etika: Tidak menggunakan scraping ilegal; data dikumpulkan secara manual dan representatif
Kode: Mengikuti standar NLP pipeline sesuai silabus
