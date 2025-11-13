# Simple Artificial Neural Network (ANN) - XOR Problem

Proyek ini merupakan implementasi sederhana jaringan syaraf tiruan (Artificial Neural Network) dari nol (tanpa library machine learning seperti TensorFlow atau PyTorch).  
Dapat digunakan untuk mempelajari cara kerja forward propagation dan backward propagation dengan tiga jenis fungsi aktivasi:

- **Sigmoid**
- **ReLU**
- **Threshold**

---

##  Fitur Utama
- Buat model ANN dengan jumlah layer sesuai keinginan.
- Pilih fungsi aktivasi untuk setiap layer.
- Latih model menggunakan dataset XOR.
- Visualisasi hasil training (error per epoch).
- Menampilkan hasil prediksi jaringan.
- Menampilkan diagram neuron (visualisasi layer dan koneksi).

---
## Cara menjalankan Program
### 1. Clone Repository
```bash
git clone https://github.com/Yazidyah/ann_project.git
cd ann_project
```
### 2. Jalankan Program
Pastikan kamu sudah berada di direktori "ann_project", lalu jalankan:
```bash
python main.py
```

### 3. Cara Penggunaan
1. Saat Program berjalan pilih "1.Buat model baru" Untuk menentukan
   - Jumlah layer
   - Jumlah neuron di setiap layer
   - Jenis aktivasi untuk tiap layer (relu, sigmoid, atau threshold)
2. Training Model
   Program akan menampilkan proses training dan menampilkan grafik error per epoch.
   Visualisasi jaringan neuron juga akan muncul (menggunakan matplotlib). 
3. Lihat Hasil Prediksi
   Setelah training selesai, hasil prediksi XOR akan ditampilkan
4. Visualisasi
   Program secara otomatis akan menampilkan
   - Grafik Error (Loss)
   - Diagram Jaringan Syaraf (Neuron & Koneksi)
  
## Dependencies
Pastikan sudah menginstal modul

    pip install matplotlib numpy


# Author
Achmad Yazid Fardin
Universitas Gunadarma
