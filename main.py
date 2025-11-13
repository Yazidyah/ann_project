import pandas as pd
import matplotlib.pyplot as plt
from network import NeuralNetwork
import visualize
import sys
import os

def load_csv_data():
    path = input("Masukkan path file CSV (misal: data.csv): ").strip()
    if not os.path.exists(path):
        print("File tidak ditemukan.")
        return None, None
    try:
        data = pd.read_csv(path)
        print(f"\nData berhasil dimuat! Kolom: {list(data.columns)}")
        print(data.head())
        input_cols = int(input("\nMasukkan jumlah kolom input: "))
        output_cols = int(input("Masukkan jumlah kolom output: "))
        inputs = data.iloc[:, :input_cols].values.tolist()
        targets = data.iloc[:, input_cols:input_cols + output_cols].values.tolist()
        return inputs, targets
    except Exception as e:
        print("Gagal memuat CSV:", e)
        return None, None

def show_structure(nn):
    print("\n=== Struktur Neural Network ===")
    print(f"Jumlah layer: {len(nn.layers)}")
    for i, layer in enumerate(nn.layers):
        print(f" Layer {i+1}: {layer.input_size} input → {layer.output_size} output | Aktivasi: {layer.activation_name}")
    print("===============================")

def train_model(nn, inputs, targets):
    print("\n Mulai training...")
    lr = float(input("Masukkan learning rate untuk training (misal 0.1): "))
    epochs = int(input("Masukkan jumlah epoch (misal 1000): "))
    errors = nn.train(inputs, targets, epochs=epochs, learning_rate=lr)
    print(" Training selesai!")

    # plot error
    plt.figure(figsize=(8,5))
    plt.plot(errors)
    plt.title("Error per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.grid(True)
    plt.show()

    # show network diagram (static)
    visualize.visualize_network(nn)

    # optional: ask to test one sample
    while True:
        t = input("Mau coba input manual untuk prediksi? (y/n): ").lower()
        if t != 'y':
            break
        vals = input("Masukkan nilai input (pisah spasi): ").strip().split()
        vals = [float(v) for v in vals]
        out = nn.forward(vals)
        print("Hasil prediksi:", out)

def main():
    nn = None
    inputs = None
    targets = None

    while True:
        print("\n=== Neural Network Interaktif ===")
        print("1. Buat model baru (tentukan layer dan aktivasi)")
        print("2. Upload data CSV")
        print("3. Lihat struktur jaringan")
        print("4. Training model + visualisasi")
        print("5. Keluar")
        choice = input("Pilih menu: ").strip()

        if choice == "1":
            print("\n🧩 Membuat model baru...")
            nn = NeuralNetwork()
            num_layers = int(input("Berapa banyak layer (layer output juga dihitung)? "))
            input_size = int(input("Masukkan jumlah neuron input: "))
            prev = input_size
            for i in range(num_layers):
                out_sz = int(input(f"  Jumlah neuron di layer {i+1}: "))
                act = input("  Fungsi aktivasi (relu/sigmoid/threshold): ").lower()
                nn.add_layer(prev, out_sz, act)
                prev = out_sz
            print(" Model dibuat.")

        elif choice == "2":
            inputs, targets = load_csv_data()
            if inputs and targets:
                print(" Data siap digunakan.")

        elif choice == "3":
            if nn:
                show_structure(nn)
            else:
                print(" Belum ada model. Buat dulu di menu 1.")

        elif choice == "4":
            if not nn:
                print(" Belum ada model. Buat dulu di menu 1.")
                continue
            if not inputs or not targets:
                yn = input("Belum ada data. Mau pakai dataset XOR otomatis? (y/n): ").lower()
                if yn == 'y':
                    inputs = [[0,0],[0,1],[1,0],[1,1]]
                    targets = [[0],[1],[1],[0]]
                    print(" Dataset XOR dimuat otomatis.")
                else:
                    print(" Upload data lewat menu 2 dulu.")
                    continue
            train_model(nn, inputs, targets)

        elif choice == "5":
            print("Keluar...")
            sys.exit()
        else:
            print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()
