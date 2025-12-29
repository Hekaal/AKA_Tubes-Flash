# Analisis Traversal Iteratif dan Rekursif pada Struktur Data Trie untuk Fitur Autocomplete

## Deskripsi Proyek
Proyek ini merupakan **Tugas Besar Mata Kuliah Analisis Kompleksitas Algoritma** yang membahas
perbandingan **algoritma traversal iteratif dan rekursif** pada **struktur data Trie**
dalam implementasi **fitur autocomplete berbasis prefix**.

Fitur autocomplete bekerja dengan mencocokkan awalan kata (*prefix*) yang diketik pengguna
dengan sekumpulan data teks, kemudian menampilkan saran kata yang relevan.
Struktur data Trie dipilih karena mampu melakukan pencarian berbasis prefix secara efisien.

Dalam proyek ini, dua pendekatan traversal Trie diimplementasikan dan dibandingkan
berdasarkan **running time**, **stabilitas performa**, serta kesesuaiannya dengan
**analisis kompleksitas Big-O**.

---

## Nama Anggota Kelompok
- Haekhal M Syaed        :103052300033
- Rizky Nur Widyatmoko   :103052300053
- Wildan Aufa Rafid      :1305213022 

---

## Dataset yang Digunakan
Dataset yang digunakan dalam proyek ini adalah **dataset nama produk e-commerce (Amazon Products)**.
Dataset ini berisi daftar nama produk dalam jumlah besar dan digunakan untuk mensimulasikan
kasus nyata fitur autocomplete pada aplikasi e-commerce.

### Informasi Dataset:
- Format: CSV
- Kolom utama: `product_name`
- Jumlah data: hingga ±1.000.000 baris
- Jenis data: teks (nama produk)

### Sumber Dataset:
Dataset diperoleh dari Kaggle dan dapat diakses melalui tautan berikut:  
🔗 https://www.kaggle.com/datasets/PromptCloudHQ/amazon-product-dataset-2020

### Penggunaan Dataset:
Nama produk pada dataset diproses dan dimasukkan ke dalam struktur data Trie.
Selanjutnya, beberapa prefix (misalnya *lug*, *tsa*, *win*, dan *blue*) digunakan
untuk menguji performa traversal iteratif dan rekursif dalam menghasilkan saran autocomplete.

---

## Ringkasan Implementasi
- Struktur data Trie digunakan untuk menyimpan data nama produk
- Dua metode traversal Trie diimplementasikan:
  - Traversal iteratif (menggunakan stack)
  - Traversal rekursif (menggunakan pemanggilan fungsi)
- Eksperimen dilakukan dengan berbagai ukuran input untuk membandingkan kinerja kedua algoritma
- Hasil eksperimen dianalisis menggunakan grafik dan dibahas dalam laporan tertulis

---

## Tujuan Akhir
Melalui proyek ini, diharapkan mahasiswa dapat:
- Memahami perbedaan implementasi algoritma iteratif dan rekursif
- Menganalisis efisiensi algoritma berdasarkan teori dan praktik
- Menyadari bahwa hasil eksperimen runtime dapat bervariasi tergantung kondisi data dan lingkungan eksekusi

---

## Catatan
Proyek ini dibuat untuk keperluan akademik dan tidak digunakan untuk tujuan komersial.
