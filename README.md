# Laporan Proyek Machine Learning - KANTO: Sistem Rekomendasi Wisata Madura Berbasis Digital

## Project Overview

Pulau Madura adalah salah satu destinasi wisata dengan potensi besar di Indonesia, menawarkan perpaduan unik antara keindahan alam, kekayaan budaya, dan sejarah yang menarik. Meskipun memiliki daya tarik yang signifikan, visibilitas destinasi wisata di Madura masih dapat ditingkatkan untuk menarik lebih banyak wisatawan. Dengan semakin beragamnya preferensi wisatawan, pengembangan sistem yang dapat memberikan rekomendasi destinasi wisata yang relevan dan sesuai dengan minat serta preferensi pengguna menjadi krusial.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi destinasi wisata di Pulau Madura yang memanfaatkan data destinasi wisata yang komprehensif. Sistem rekomendasi ini dirancang untuk membantu wisatawan dalam memilih tempat wisata berdasarkan informasi detail seperti nama wisata, asal kota, alamat lengkap, jam buka, jam tutup, deskripsi wisata, kategori wisata, jumlah pengunjung, harga tiket, fasilitas, rating, serta koordinat geografis (latitude dan longitude). Dengan sistem rekomendasi yang cerdas ini, wisatawan tidak hanya mendapatkan informasi yang sesuai dengan preferensi pribadi mereka, tetapi juga dapat mengoptimalkan pengalaman berwisata mereka di Madura dengan memilih tempat yang menawarkan nilai terbaik sesuai dengan harapan.

Solusi yang Ditawarkan:
Sistem rekomendasi yang dikembangkan dalam proyek ini menggunakan pendekatan machine learning dengan data destinasi wisata Madura untuk memberikan rekomendasi yang lebih terarah dan relevan bagi wisatawan. Sistem ini akan mempertimbangkan berbagai faktor, seperti rating tempat wisata, ketersediaan fasilitas, jumlah pengunjung, harga tiket, dan lokasi geografis. Model rekomendasi dibangun menggunakan TensorFlow dan Keras dengan arsitektur Multilayer Perceptron (MLP). Data akan melalui pra-pemrosesan seperti penggunaan TF-IDF untuk fitur teks (kota), normalisasi untuk fitur numerik (rating, fasilitas, jumlah pengunjung, harga tiket), serta Label Encoder untuk kategori wisata. Pendekatan ini bertujuan untuk memberikan rekomendasi yang lebih personal dan meningkatkan visibilitas destinasi wisata di Madura.

Daftar referensi:
[1]	Faurina, R., & Sitanggang, E. (2023). Implementasi Metode Content-Based Filtering dan Collaborative Filtering pada Sistem Rekomendasi Wisata di Bali. Techno.COM, 22(4), 870-881.
[2]	Firmansyah, M. F., Aziz, A., & Ahsan, M. (2024). Peningkatan Kinerja Sistem Rekomendasi Wisata Melalui Penerapan Algoritma Collaborative Filtering dan K-Nearest Neighbors dengan Metode Klasterisasi K-Means. JATI (Jurnal Mahasiswa Teknik Informatika), 8(6), 11420-11425.
[3]	Siska, S., Fajri, I. N., Rayhan, R., Pratama, A., & Rohman, A. N. (2024). Sistem Rekomendasi Wisata Magelang Menggunakan Metode Collaborative Filtering. Jurnal Eksplora Informatika, 14(1), 63-68.

Bagian laporan ini mencakup:
## Business Understanding

Pada bagian ini, akan dijelaskan proses klarifikasi masalah dengan proyek rekomendasi destinasi wisata di Pulau Madura. Berdasarkan proyek yang diangkat, tujuan utama adalah membangun sistem yang dapat memberikan rekomendasi destinasi wisata yang relevan dan sesuai dengan preferensi pengguna di Madura. Sistem ini mempertimbangkan faktor-faktor penting yang terdapat dalam dataset, seperti rating, kategori destinasi wisata, harga tiket, fasilitas, dan jumlah pengunjung.

### Problem Statements

Pernyataan masalah yang ada dalam proyek ini adalah sebagai berikut:
1. Banyak wisatawan mengalami kesulitan dalam memilih destinasi wisata yang sesuai dengan preferensi mereka di Pulau Madura karena terdapat begitu banyak pilihan destinasi yang ada dan informasi yang tersebar. Hal ini dapat menyebabkan wisatawan kehilangan waktu dan potensi pengalaman berwisata yang optimal.
2. Belum adanya sistem yang secara efektif memanfaatkan data detail destinasi (seperti rating, fasilitas, deskripsi, dan kategori) untuk memberikan rekomendasi yang personal dan terarah. Wisatawan sering kali bergantung pada sumber informasi yang terbatas dan kurang tepat untuk memilih destinasi yang sesuai dengan keinginan mereka.
3. Meskipun Madura memiliki beragam jenis destinasi wisata (alam, budaya, edukasi, kebun, kuliner, pantai, religi taman, wahana air), banyak di antaranya mungkin kurang terekspos atau diketahui oleh wisatawan. Tanpa sistem rekomendasi yang relevan dan terperinci, destinasi-destinasi potensial ini sulit dijangkau oleh target wisatawan yang sesuai.

### Goals

Berdasarkan pernyataan masalah yang ada, berikut adalah tujuan proyek yang diharapkan dapat menyelesaikan masalah-masalah tersebut:
1. Membangun sebuah sistem rekomendasi destinasi wisata di Madura yang personal dan relevan berdasarkan informasi detail destinasi (seperti rating, kategori, fasilitas, dan harga tiket), untuk memudahkan wisatawan dalam memilih destinasi wisata sesuai dengan preferensi pribadi mereka.
2. Menggunakan data komprehensif destinasi wisata (nama wisata, asal kota, alamat lengkap, jam buka, jam tutup, deskripsi, kategori, jumlah pengunjung, harga tiket, fasilitas, rating, latitude, longitude) untuk memberikan rekomendasi destinasi yang lebih personal dan akurat. Sistem rekomendasi ini akan memanfaatkan algoritma berbasis machine learning untuk menghasilkan rekomendasi yang sesuai dengan keinginan dan minat pengguna.
3. Menyusun sistem rekomendasi yang dapat menyesuaikan dengan berbagai jenis destinasi wisata di Madura, sehingga wisatawan dapat dengan mudah menemukan destinasi yang sesuai dengan minat mereka berdasarkan analisis data, sekaligus membantu meningkatkan visibilitas dan kunjungan ke destinasi-destinasi tersebut.

### Solution Approach

Untuk mencapai tujuan yang telah ditetapkan, pendekatan machine learning yang digunakan adalah pembangunan model klasifikasi untuk kategori destinasi wisata. Berikut adalah pendekatan yang akan digunakan dalam proyek ini:
* Proyek ini menggunakan model Multilayer Perceptron (MLP) yang diimplementasikan dengan TensorFlow dan Keras. Model ini dipilih untuk melakukan klasifikasi kategori destinasi wisata berdasarkan fitur-fitur yang relevan.
* Model ini akan memproses fitur teks (kolom "asal kabupaten") menggunakan TF-IDF, menormalisasi fitur numerik (rating, fasilitas, jumlah pengunjung, harga tiket), dan menggunakan Label Encoder untuk kategori wisata. Proses ini memungkinkan model untuk mempelajari pola dan hubungan antar fitur dalam dataset destinasi.
* Model yang telah dilatih akan digunakan untuk memprediksi atau mengklasifikasikan kategori destinasi berdasarkan input yang diberikan (seperti nama kota dan fitur numerik lainnya).

#### Proses:
1.  Data destinasi wisata akan melalui tahap pra-pemrosesan yaitu Pengolahan fitur teks menggunakan TF-IDF, Normalisasi fitur numerik, dan Penggunaan Label Encoder untuk fitur kategorikal.
2.  Model MLP akan dilatih menggunakan data yang telah diproses. Kinerja model akan dievaluasi menggunakan metrik akurasi dan loss selama pelatihan dan fine-tuning.
3.  Model yang telah terlatih akan disimpan dalam format (.h5) untuk memudahkan implementasi dan inferensi di backend.
4.  Melakukan Prediksi dan Rekomendasi, sistem akan memberikan rekomendasi destinasi yang relevan sesuai dengan preferensi atau kriteria yang dicari.
      
## Data Understanding
Pada bagian ini, akan dijelaskan dataset yang digunakan dalam proyek sistem rekomendasi destinasi wisata di Pulau Madura. Dataset ini berisi informasi tentang destinasi wisata di Madura, dengan kolom-kolom sebagai berikut:

  * nama_wisata: Nama dari destinasi wisata.
  * asal_kota: Kota asal destinasi wisata.
  * alamat_lengkap: Alamat lengkap destinasi wisata.
  * jam_buka: Jam operasional buka destinasi.
  * jam_tutup: Jam operasional tutup destinasi.
  * deskripsi_wisata: Deskripsi singkat atau informasi tentang destinasi wisata.
  * kategori_wisata: Kategori dari destinasi wisata.
  * jumlah_pengunjung: Estimasi jumlah pengunjung.
  * harga_tiket: Harga tiket atau biaya masuk destinasi wisata.
  * fasilitas: Informasi fasilitas yang tersedia di destinasi wisata.
  * rating: Rating keseluruhan destinasi wisata.
  * lat: Koordinat geografis destinasi yang menunjukkan posisi utara atau selatan dari garis khatulistiwa (lintang).
  * lon: Koordinat geografis destinasi yang menunjukkan posisi timur atau barat dari garis meridian utama (bujur).

### Jumlah Total Baris Data:
103 baris data

### Kondisi Data:
  * Tidak ada data missing value.
  * Tidak ada data duplikat yang terdeteksi.

### Distribusi Data:
  * Asal Kota: Distribusi destinasi tersebar di berbagai kota di Madura (Bangkalan, Sumenep, Pamekasan, Sampang). Hal ini penting untuk rekomendasi berdasarkan lokasi.
  * Kategori Wisata: Terdapat berbagai kategori wisata seperti alam, budaya, edukasi, kebun, kuliner, pantai, religi taman, wahana air, menunjukkan keberagaman jenis destinasi.
  * Harga Tiket: Distribusi harga tiket bervariasi, dari gratis hingga berbayar, mencerminkan opsi yang beragam bagi wisatawan.
  * Rating: Rating destinasi bervariasi, mencerminkan kualitas atau popularitas destinasi dari perspektif pengunjung.
  * Jumlah Pengunjung: Distribusi jumlah pengunjung juga bervariasi, memberikan gambaran tentang popularitas destinasi.

### Masalah Umum pada Data:
  * Variasi Format: Beberapa kolom mungkin memiliki variasi format yang memerlukan standardisasi (misalnya, untuk harga tiket jika masih mengandung simbol mata uang).

### Exploratory data analysis:
Berdasarkan hasil eksplorasi data yang telah dilakukan, beberapa insight yang dapat diperoleh adalah:
  * Asal Kota Destinasi: Mengetahui distribusi destinasi berdasarkan asal kota dapat membantu dalam strategi rekomendasi berbasis lokasi.
  * Distribusi Harga Tiket: Rentang harga tiket sangat bervariasi, menunjukkan bahwa Madura menawarkan destinasi wisata dengan berbagai tingkat harga, dari gratis hingga berbayar.
  * Distribusi Jumlah Pengunjung: Gambaran tentang jumlah pengunjung dapat mengindikasikan popularitas destinasi tertentu.
  * Distribusi Rating: Sebagian besar destinasi memiliki rating tertentu, yang memberikan gambaran umum tentang kepuasan pengunjung.
  
### Sumber dataset
Sumber data yang digunakan berasal dari dataset destinasi wisata yang dapat diakses melalui tautan berikut: 
[https://docs.google.com/spreadsheets/d/1FYtbusduf4XeDl9eFYmtwFVKesOkmmqD/edit?usp=sharing&ouid=109668250720515860025&rtpof=true&sd=true](https://docs.google.com/spreadsheets/d/1FYtbusduf4XeDl9eFYmtwFVKesOkmmqD/edit?usp=sharing&ouid=109668250720515860025&rtpof=true&sd=true)

## Data Preparation
Berikut adalah teknik yang digunakan untuk mempersiapkan data dalam proyek sistem rekomendasi destinasi wisata di Madura:
  * Encoding:
      * Kolom `kategori_wisata` akan di-encode menggunakan Label Encoder.
      * Kolom `asal_kota` akan diolah menggunakan TF-IDF.
  * Normalisasi
      * Kolom `rating`, `fasilitas`, `jumlah pengunjung`, dan `harga tiket` akan dinormalisasi.

## Modeling
Tahapan Modeling ini membahas mengenai pembangunan sistem rekomendasi yang digunakan untuk memberikan solusi terhadap permasalahan dalam memilih destinasi wisata yang sesuai dengan preferensi pengguna di Pulau Madura. Berbeda dengan pendekatan rekomendasi tradisional berbasis Collaborative Filtering atau Matrix Factorization, model yang saya kembangkan berfokus pada klasifikasi kategori destinasi wisata menggunakan arsitektur Multilayer Perceptron (MLP). Pendekatan ini memungkinkan sistem untuk mengidentifikasi jenis destinasi wisata yang paling mungkin disukai pengguna berdasarkan input tertentu, dan kemudian merekomendasikan destinasi yang relevan dari kategori tersebut.

A. Model Klasifikasi dengan Multilayer Perceptron (MLP)
  Model yang digunakan adalah Multilayer Perceptron (MLP) yang diimplementasikan menggunakan TensorFlow dan Keras. MLP adalah jenis feedforward artificial neural network yang efektif untuk tugas klasifikasi. Dalam konteks ini, model belajar memetakan fitur-fitur destinasi wisata (seperti kota asal, rating, fasilitas, jumlah pengunjung, dan harga tiket) ke dalam kategori wisata yang sesuai.

  * Arsitektur dan Proses Model MLP:
    * Input Layer: Menerima input dari fitur yang telah dipra-proses. Ini mencakup representasi TF-IDF dari kolom asal_kota (untuk memahami karakteristik kota asal) dan fitur numerik yang telah dinormalisasi (rating, fasilitas, jumlah_pengunjung, harga_tiket).
    * Hidden Layers: Model ini memiliki beberapa dense layer (lapisan terhubung penuh) dengan fungsi aktivasi ReLU (relu). Lapisan-lapisan ini bertugas mengekstraksi pola dan hubungan non-linear yang kompleks dari data input. Contohnya, notebook menunjukkan penggunaan dense layer dengan 128 unit, diikuti oleh 64 unit.
    * Output Layer: Merupakan dense layer terakhir yang memiliki jumlah unit sama dengan jumlah kategori wisata unik yang ada dalam dataset. Fungsi aktivasi yang digunakan adalah softmax (softmax), yang menghasilkan distribusi probabilitas di atas semua kategori. Probabilitas tertinggi menunjukkan kategori yang paling mungkin untuk destinasi dengan fitur input tersebut.
    * Kompilasi Model: Model dikompilasi dengan optimizer Adam (efisien untuk pelatihan neural network), loss function sparse_categorical_crossentropy (cocok untuk klasifikasi multi-kelas dengan label integer), dan metrik accuracy untuk memantau kinerja.
    
  * Kelebihan Model MLP untuk Klasifikasi Kategori:
    * MLP dapat menangkap pola dan interaksi kompleks antar fitur yang mungkin tidak linier, sehingga menghasilkan klasifikasi kategori yang lebih akurat.
    * Model dapat beradaptasi dengan berbagai jenis input (numerik dan hasil vektorisasi teks) setelah pra-pemrosesan yang tepat.
    * Setelah dilatih, model MLP dapat memberikan prediksi kategori dengan sangat cepat, yang penting untuk sistem rekomendasi real-time.
    * Karena model mengklasifikasikan destinasi berdasarkan karakteristiknya (bukan interaksi pengguna-item), ia dapat merekomendasikan destinasi baru atau kepada pengguna baru tanpa riwayat interaksi sebelumnya.
    
  * Kekurangan Model MLP untuk Klasifikasi Kategori:
    * Model ini memerlukan label kategori yang jelas dan akurat untuk setiap destinasi selama pelatihan.
    Tidak Langsung Memberikan Rekomendasi Item: Model ini hanya mengklasifikasikan kategori destinasi. Untuk memberikan rekomendasi item spesifik (nama wisata), dibutuhkan langkah tambahan yaitu memfilter dataset berdasarkan kategori yang diprediksi.
    * Jika arsitektur model terlalu kompleks atau data pelatihan terbatas, model dapat mengalami overfitting, mengurangi kemampuan generalisasinya pada data baru.

B. Top-N Recommendation
Setelah model MLP berhasil dilatih dan disimpan, tahap selanjutnya adalah menggunakannya untuk memberikan rekomendasi kepada wisatawan. Proses ini melibatkan prediksi kategori dan kemudian penyaringan destinasi.
  * Proses Prediksi:
    * Input Pengguna: Sistem akan menerima input dari pengguna, yang dapat berupa preferensi kota (asal_kota), atau karakteristik destinasi yang dicari (rating, fasilitas, jumlah_pengunjung, harga_tiket).
    * Pra-pemrosesan Input: Input ini akan melalui tahap pra-pemrosesan yang sama seperti data pelatihan, yaitu:
    Teks asal_kota diubah menjadi vektor TF-IDF menggunakan TfidfVectorizer yang sama yang dilatih sebelumnya.
    Fitur numerik dinormalisasi menggunakan MinMaxScaler yang sama yang dilatih sebelumnya.
    * Inferensi Model: Fitur input yang sudah dipra-proses kemudian dimasukkan ke dalam model MLP yang telah dilatih. Model akan menghasilkan probabilitas untuk setiap kategori wisata.
    * Penentuan Kategori Prediksi: Kategori dengan probabilitas tertinggi akan dipilih sebagai kategori destinasi yang diprediksi sesuai dengan preferensi input pengguna.
    * Dekode Kategori: Label kategori yang telah di-encode akan diubah kembali menjadi nama kategori yang dapat dimengerti (misalnya, dari 0 menjadi Wisata Alam) menggunakan LabelEncoder yang sama.
    
  * Penyajian Top-N Recommendation:
    Setelah kategori destinasi diprediksi, langkah selanjutnya adalah menyajikan rekomendasi destinasi spesifik kepada pengguna.
    * Dataset asli destinasi wisata Madura akan difilter untuk menampilkan hanya destinasi yang termasuk dalam kategori yang diprediksi.
    * Destinasi dalam kategori yang diprediksi dapat diurutkan berdasarkan kriteria tambahan seperti rating (tertinggi), jumlah_pengunjung, atau harga_tiket (terendah) untuk memberikan rekomendasi yang lebih optimal.
    * Sistem akan menyajikan Top-N (dalam kasus ini, Top 10) destinasi teratas dari hasil penyaringan dan pengurutan kepada pengguna. Ini memberikan daftar rekomendasi yang relevan dan terkurasi.

  * Kelebihan Top-N Recommendation sebagai Output:
    * Memberikan daftar yang terstruktur dan mudah dipahami bagi pengguna.
    * Pengguna tidak perlu menelusuri seluruh dataset; mereka langsung disajikan pilihan terbaik.
    * Rekomendasi didasarkan pada kategori yang diprediksi sesuai preferensi input pengguna.
  
  * Kekurangan Top-N Recommendation:
    * Jika prediksi kategori oleh model kurang akurat, rekomendasi Top-N juga akan kurang relevan.
    * Pendekatan ini mungkin membatasi pengguna pada destinasi dalam kategori yang diprediksi, mengurangi peluang untuk menemukan destinasi menarik di luar kategori tersebut.

Hasil Prediksi menggunakan Top-N Recommendation

![gambar 5](https://github.com/user-attachments/assets/be1f1784-d64f-44f1-982e-65152ce80590)


## Evaluation
Pada bagian Evaluasi, akan dijelaskan metrik evaluasi yang digunakan untuk menilai kinerja model sistem rekomendasi destinasi wisata di Pulau Madura. Karena model yang dibangun adalah model klasifikasi Multilayer Perceptron (MLP) untuk memprediksi kategori wisata, metrik evaluasi yang relevan adalah Akurasi dan Loss (Sparse Categorical Crossentropy).

A. Akurasi (Accuracy)
Akurasi adalah metrik evaluasi utama yang digunakan untuk mengukur kinerja model klasifikasi. Akurasi menghitung proporsi prediksi yang benar (yaitu, berapa banyak kategori destinasi wisata yang diprediksi dengan benar oleh model) dari total jumlah prediksi. Alasan penggunaan Accuracy karena merupakan metrik langsung dan mudah dipahami untuk mengukur persentase prediksi kategori yang benar oleh model.

B. Sparse Categorical Crossentropy (Loss Function)
Sebagai fungsi loss dalam proses pelatihan model klasifikasi MLP, digunakan Sparse Categorical Crossentropy. Fungsi loss ini mengukur "kesalahan" atau "jarak" antara probabilitas kategori yang diprediksi oleh model dengan label kategori yang sebenarnya (dalam format integer, seperti yang dihasilkan oleh Label Encoder). Model berusaha meminimalkan nilai loss ini selama pelatihan. Alasan penggunaan Loss Function karena model mengklasifikasikan ke dalam banyak kategori (multi-kelas) dan label kategori telah diubah menjadi format integer (setelah Label Encoding), membuatnya efisien dan tepat untuk melatih neural network pada tugas ini.

Berikut adalah hasil evaluasi model:

| Epoch | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| 30 | 1.00 | 1.00 | 0.0041 | 5.06 |

![gambar 6](https://github.com/user-attachments/assets/d6f1a0ad-c773-491a-be6f-fa803e697d21)
