# Laporan Proyek Machine Learning - Fathia Rahma
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Ekonomi dan bisnis**, dengan judul **Predictive Analytics: Harga Rumah**  

### Latar Belakang

Pasar properti memiliki peranan penting dalam kestabilan ekonomi, baik di tingkat regional maupun nasional. Harga rumah yang dinamis dan dipengaruhi oleh berbagai faktor seperti lokasi, ukuran bangunan, usia properti, serta kondisi ekonomi makro, menjadikan prediksi harga rumah sebagai tantangan tersendiri. Ketidakpastian harga ini dapat mempersulit pembeli dalam merencanakan keuangan, menyulitkan penjual dalam menentukan harga jual yang optimal, dan menyulitkan pihak perbankan atau lembaga pembiayaan dalam menilai risiko kredit properti.

Fluktuasi harga rumah yang tidak dapat diprediksi secara akurat dapat menyebabkan inefisiensi pasar, menghambat alokasi sumber daya secara optimal, dan berdampak pada meningkatnya risiko dalam investasi properti [[1](https://arxiv.org/abs/2110.07151)]. Hal ini terutama berdampak signifikan di daerah urban dengan pertumbuhan penduduk tinggi dan tingkat permintaan perumahan yang terus meningkat.

Selain itu, metode penilaian harga rumah secara konvensional seperti metode perbandingan pasar (market comparison approach) bersifat subjektif dan memerlukan waktu serta biaya yang tidak sedikit. Hal ini menjadikan pendekatan berbasis data menggunakan machine learning sebagai alternatif yang efisien dan objektif. Algoritma machine learning seperti Random Forest dan Gradient Boosting telah terbukti mampu memberikan estimasi harga yang lebih akurat dengan memanfaatkan data historis dan multivariat [[2](https://arxiv.org/abs/2504.04303)] [[3](https://arxiv.org/abs/2006.10092)].

Dengan mengembangkan sistem prediksi harga rumah berbasis machine learning, kita dapat membantu berbagai pihak-mulai dari calon pembeli, penjual, pengembang properti, hingga lembaga keuangan — untuk mengambil keputusan yang lebih tepat dan berdasarkan data. Implementasi solusi ini juga dapat meningkatkan efisiensi pasar dan mendukung pengambilan keputusan strategis di sektor perumahan.


## Business Understanding
Perkembangan sektor properti di berbagai wilayah, memunculkan kebutuhan akan sistem pendukung keputusan yang dapat membantu memperkirakan harga rumah secara lebih akurat. Informasi mengenai estimasi harga rumah sangat berguna bagi calon pembeli, penjual, pengembang properti, hingga lembaga keuangan seperti bank. Memprediksi harga rumah secara akurat dapat meningkatkan efisiensi pasar, mempercepat proses jual-beli, serta mengurangi risiko dalam pengambilan keputusan. Contoh nyata dari manfaat prediksi harga rumah ini misalnya membantu bank dalam menilai jaminan pinjaman KPR, membantu penjual dalam menentukan harga optimal, atau mendukung kebijakan pemerintah dalam perencanaan tata ruang.
### Problem Statements
Berdasarkan latar belakang di atas, berikut ini rincian masalah yang dapat diselesaikan dalam proyek ini:
-  Bagaimana membangun model machine learning yang mampu memprediksi harga rumah berdasarkan fitur-fitur properti seperti luas bangunan, jumlah kamar, dan fasilitas lainnya?
-  Model seperti apa yang mampu memberikan prediksi harga paling akurat?
-  Fitur-fitur mana saja yang paling memengaruhi harga rumah?

### Goals
Tujuan dari proyek ini adalah:
- Mengembangkan model prediksi harga rumah menggunakan teknik machine learning.
- Membandingkan beberapa algoritma untuk mendapatkan model dengan performa terbaik.
- Mengidentifikasi fitur-fitur penting yang berkontribusi besar terhadap harga rumah.

### Solution Statements
-  Melakukan analisis data dengan pendekatan eksplorasi data (EDA), baik univariat maupun multivariat, untuk memahami distribusi data dan hubungan antar fitur.
- Membuat beberapa variasi model untuk mendapatkan model terbaik dari beberapa model yang telah dibuat untuk prediksi harga rumah. Diantaranya adalah menggunakan KNN, Random Forest, SVM, Gradient Boosting Regressor dan Linear Regression
- Menggunakan _Mean Absolute Error_ (MAE), _Root Mean Square Error_ (RMSE) dan _R Squared_ untuk mengevaluasi model

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**


| Jenis | Keterangan |
| ------ | ------ |
| Title | _Housing Price Prediction_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction) |
| Maintainer | [Harish Kumar ⚡](https://www.kaggle.com/harishkumardatalab) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | _Data Visualization, Classification, Real Estate, Categorical_ |
| Usability | 10.00 |

Berikut informasi pada dataset: 
Dataset yang digunakan dalam proyek ini adalah data harga rumah yang berasal dari situs Kaggle dengan judul “_Housing Price Prediction_”, disediakan oleh Harish Kumar DataLab. Dataset ini merupakan data publik yang berisi informasi mengenai berbagai atribut rumah dan harga jualnya.

| price | area | bedrooms | bathrooms | stories | mainroad | guestroom | basement | hotwaterheating | airconditioning | airconditioning | airconditioning | airconditioning |
| ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ |------ | ------ |------ |------ |
| 13300000 | 7420 | 4 | 2 | 3 | yes | no	| no  | no | yes |  2  | yes |furnished |
| 12250000 | 8960 | 4 | 4 | 4 | yes | no	| no  | no | yes |  3  | no  |furnished |
| 12250000 | 9960 |	3 | 2 | 2 | yes | no	| yes | no | no  |  2  | yes |semi-furnished |
| 12215000 | 7500 | 4 | 2 | 2 | yes | no	| yes | no | yes |  3  | yes |furnished |
| 11410000 | 7420 | 4 | 1 | 2 | yes | yes	| yes | no | yes |  2  | no  |furnished |

Tabel 1. Isi Dataset

Dilihat dari _Tabel 1. Isi Dataset_ dapat disimpulkan bahwa:
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 545  sample dengan 13 fitur.
- Dataset memiliki 6 fitur bertipe float64 dan 7 fitur bertipe object.
- Terdapat 1 missing value dalam dataset.

### Variable dalam dataset
| Nama Variabel | Tipe Data   | Deskripsi |
|---------------|-------------|-----------|
| `price`       | Numerik     | Harga rumah.                 |
| `area`        | Numerik     | Luas rumah dalam satuan kaki persegi. |
| `bedrooms`    | Numerik     | Jumlah kamar tidur.            |
| `bathrooms`   | Numerik     | Jumlah kamar mandi.         |
| `stories`     | Numerik     | Jumlah lantai (tingkat) rumah.       |
| `mainroad`    | Kategorikal | Apakah rumah terhubung ke jalan utama (`Yes` / `No`). |
| `guestroom`   | Kategorikal | Apakah rumah memiliki kamar tamu (`Yes` / `No`). |
| `basement`    | Kategorikal | Apakah rumah memiliki ruang bawah tanah (`Yes` / `No`). |
| `hotwaterheating`| Kategorikal | Apakah rumah memiliki sistem pemanas air (`Yes` / `No`).                  |
| `airconditioning`| Kategorikal | Apakah rumah memiliki AC (`Yes` / `No`).          |
| `parking`     | Numerik     | Jumlah tempat parkir.         |
| `prefarea`    | Kategorikal | Apakah rumah berada di area yang diinginkan (`Yes` / `No`). |
| `furnishingstatus`  | Kategorikal | Status perabotan rumah (`furnished`, `semi-furnished`, `unfurnished`).   |

Tabel 2. Variabel dalam Dataset

### EDA - Data Cleaning

Setelah dilakukan pengecekan _missing value_ dan duplikasi data, dataset ini sudah cukup bersih dan tidak mengandung nilai hilang, sehingga dapat langsung dilakukan eksplorasi

![Outlier (Data Numerik)](https://i.ibb.co/Z6d6RQgY/download.png)

Gambar 1. Outlier Data Numerik 

Berdasarkan boxplot pada Gambar 1., atribut `price`  dan `area`  menunjukkan adanya outlier. Namun, setelah dilakukan pengecekan manual terhadap nilai-nilai tersebut, ditemukan bahwa properti dengan harga dan luas besar ini memiliki fitur-fitur yang sesuai, seperti jumlah kamar lebih banyak, berperabotan lengkap, atau terletak di area premium. Oleh karena itu, outlier ini dianggap sebagai data valid dan tetap dipertahankan dalam proses analisis. Selain itu outlier pada atribut `bedrooms`, `bathrooms`, `stories`, `parking` juga masih realistis dengan mempertimbangkan nilai dari atribut lainnya.


### EDA - Univariate Analysis

![Analisis Univariat (Data Kategori)](https://i.ibb.co/jZD0VRkV/download.png)

Gambar 2a. Analisis Univariat (Data Kategori) 

![Univariate Analysis](https://i.ibb.co/V2mQ2dK/EDA-Univariate.png)

Gambar 2b. Analisis Univariat (Data Numerik) 

 Berdasarkan _Gambar 2a_ , dapat dilihat bahwa:
 1. Terdapat 3 kategori pada fitur `furnishingstatus`, secara berurutan dari jumlahnya yang paling banyak yaitu: semi-furnished, unfurnished dan furnished. Dari data persentase dapat kita simpulkan bahwa lebih dari 42% sampel masih berupa semi-furnished.
 2. Fitur `mainroad` didominasi oleh properti yang memiliki akses ke jalan utama (yes)
 3. Fitur `guestroom`, `hotwaterheating`, `prefarea`, `basement `, dan `airconditioning` menunjukkan mayoritas properti tidak memiliki fasilitas tersebut.

 Pada _Gambar 2b_, untuk data numerik memiliki karakteristik, yaitu:
- `price` & `area`: Distribusi miring ke kanan (right-skewed), menunjukkan mayoritas properti bernilai rendah hingga menengah, dan hanya sedikit yang sangat mahal atau luas.
- `bedrooms` & `bathrooms`: Rumah dengan 3 kamar tidur dan 1 kamar mandi paling umum
- `parking`: Sebagian besar properti tidak memiliki tempat parkir


### EDA - Multivariate Analysis

![Multivariate Analysis](https://i.ibb.co.com/GQQBPZps/Untitled.png)
Gambar 3a. Analisis Multivariat Kategorikal

Pada _Gambar 3a_ memperlihatkan rata-rata harga relatif terhadap fitur-fitur kategori, yang dapat ditarik kesimpulan bahwa semakin tinggi harga barang artinya semakin banyak fasilitas yang tersedia, contohnya pada ftur basement, mainroad dll.

![Multivariate Analysis](https://i.ibb.co.com/ynjWtbD3/download.png)
Gambar 3b. Analisis Multivariat Numerik

![Multivariate Analysis](https://i.ibb.co.com/YBvvs6Bt/download.png)
Gambar 3c. Analisis Matriks Korelasi

Analisis Multivariat_, dengan menggunakan fungsi _pairplot_ dari _library seaborn_, tampak terlihat relasi pasangan dalam dataset menunjukan pola acak. Pada _Gambar 3b_ fitur `area` berkorelasi positif dengan fitur `price` yang artinya semakin tinngi nilai `area` maka berbanding lurus dengan nilai `price`. Analisis Matriks Korelasi_, merupakan _Correlation Matrix_ menunjukkan hubungan antar fitur dalam nilai korelasi. Jika diamati, fitur  `area` memiliki skor korelasi yang cukup besar `0.54` dengan fitur target `price` .

## Data Preparation
 Data preparation merupakan tahapan penting di mana data dilakukan proses transformasi sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahapan yang umum dilakukan pada data preparation, antara lain, seleksi fitur, transformasi data, feature engineering, dan dimensionality reduction. 

Pada bagian ini akan dilakukan tahap persiapan data, sebagai berikut:
- *Data Cleaning*
Di tahap EDA dataset telah diperiksa dan tidak ditemukan missing value, sehingga tidak perlu penanganan

- Encoding fitur kategori.
Encoding dilakukan untuk untuk membantu model memahami dalam representasi numerik. Variabel kategori seperti `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, dan `prefarea` diubah dari bentuk teks ("yes"/"no") menjadi numerik biner (1/0). Sedangkan variabel `furnishingstatus` yang memiliki tiga kategori (furnished, semi-furnished, unfurnished) diencoding menjadi angka 1, 2, dan 3 sesuai urutan kategorinya

- Pembagian dataset
Dataset dibagi menjadi data training dan testing dengan proporsi 80:20. Pembagian dilakukan secara acak namun reproducible dengan menetapkan random state. Hal ini untuk memastikan model dapat diuji pada data yang belum pernah dilihat sebelumnya.

- Standarisasi.
Semua fitur numerik diskalakan menggunakan StandardScaler, yaitu mengubah data sehingga memiliki distribusi dengan rata-rata (mean) nol dan standar deviasi satu. Hal ini penting, agar fitur yang memiliki rentang nilai besar tidak mendominasi model.


## Modeling
Algoritma pada proyek ini melakukan pemodelan dengan 5 algoritma, yaitu:

### _K-Nearest Neighbors (KNN)_
algoritma machine learning yang bekerja dengan menemukan k tetangga terdekat dari data baru dan kemudian menggunakan kategori atau nilai rata-rata dari tetangga tersebut untuk memprediksi nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
-  `n_neighbors` jumlah tetangga terdekat.
- `weight = distance` Tetangga yang lebih dekat memiliki pengaruh lebih besar.

Keunggulan _KNN_ :
- Dapat digunakan untuk klasifikasi dan regresi.
- Sederhana dan mudah dipahami.

Kekurangan _KNN_ :
- Sensitif terhadap skala fitur dan outlier. 
- Membutuhkan banyak memori dan waktu komputasi untuk dataset besar. 

### _Random Forest_
algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
- `max_depth` kedalaman maksimum.

Keunggulan _Random Forest_ :
- Mengurangi overfitting dibanding decision tree tunggal.
- Tahan terhadap outlier dan fitur non-linear.

Kekurangan _Random Forest_ :
- Sulit untuk interpretasi model.
- Memerlukan sumber daya komputasi lebih besar.

### _Support Vector Machine (SVM)_
algoritma machine learning yang digunakan bekerja dengan menemukan hyperplane terbaik yang memaksimalkan margin antara data dalam ruang fitur. Untuk regresi digunakan SVR dengan kernel RBF. Parameter default digunakan dalam proyek ini.
 
Keunggulan _Support Vector Machine (SVM)_ :
- Efektif pada data berdimensi tinggi.
- Memiliki kemampuan kernel untuk menangani non-linearitas.

Kekurangan  _Support Vector Machine (SVM)_ :
- Sensitif terhadap pemilihan kernel dan parameter.
- Memerlukan waktu pelatihan yang cukup lama pada dataset besar.

### _Linear Regression_ 
Model regresi linier yang mengasumsikan hubungan linear antara variabel independen dan dependen. Model ini berusaha meminimalkan jumlah kuadrat kesalahan prediksi.
 
Keunggulan _Linear Regression_:
- Mudah diinterpretasikan.
- Cepat untuk dilatih dan diprediksi

Kekurangan _Linear Regression_:
- Tidak cocok untuk hubungan data yang non-linear.
- Rentan terhadap multikolinearitas antar fitur.

### _Gradient Boosting_ 
Algoritma ensemble yang membangun model secara bertahap dengan memperbaiki kesalahan model sebelumnya menggunakan boosting.

keuntungan _Gradient Boosting_ :
- Akurasi tinggi pada berbagai masalah regresi.
- Fleksibel untuk berbagai fungsi loss dan parameter.

Kerugian _Gradient Boosting_ :
- Rentan terhadap overfitting jika tidak diatur dengan baik.
- Proses pelatihan lebih lambat dibanding Random Forest.



## Evaluation

Dalam tahap evaluasi, metrik yang digunakan adalah `MAE`, `RMSE` dan `R² Score`

### 1. **Mean Absolute Error (MAE)**
MAE mengukur rata-rata selisih absolut antara nilai aktual dan prediksi. Metrik ini memberikan gambaran langsung seberapa besar kesalahan model secara umum. Semakin kecil MAE, semakin baik model dalam memprediksi.
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$

### 2. **Root Mean Squared Error (RMSE)**
RMSE mengukur akar dari rata-rata kuadrat kesalahan. Metrik ini memberikan penalti lebih besar terhadap kesalahan yang besar. RMSE yang rendah menandakan prediksi model mendekati nilai aktual secara konsisten.
$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 }
$$

### 3. **R-squared (R² Score)**
R² mengukur proporsi variansi target yang bisa dijelaskan oleh model. Skor berkisar dari 0 hingga 1.
$$
R^2 = 1 - \frac{ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 }{ \sum_{i=1}^{n} (y_i - \bar{y})^2 }
$$

- Nilai R² mendekati 1 → model sangat baik menjelaskan variansi data
- Nilai R² < 0.5 → model masih kurang baik dalam menjelaskan variasi data.

Berikut hasil evaluasi dari model yang dilatih:

| Model | MAE | RMSE | R² Score |
| ------ | ------ | ------ | ------ |
| Gradient Boosting | 0.083555 | 0.112721 | 0.664659 |
| Linear Regression  | 0.084821	| 0.115244 | 0.649475 |
| SVR (RBF Kernel) | 0.086523 | 0.115991 | 0.644917 |
| Random Forest | 0.088176 | 0.121048	 | 0.613282 |
| KNN Regressor | 0.096125 | 0.137827 | 0.498643 |

Tabel 3. Hasil Evaluasi

![Plot Evaluasi](https://i.ibb.co.com/jvWt9thW/download.png)

Gambar 3. Visualisasi Actual vs Predicted

Berdasarkan Tabel 3 dan Gambar 3, dapat disimpulkan bahwa model dengan performa terbaik adalah Gradient Boosting Regressor, karena memiliki nilai MAE dan RMSE yang paling rendah, serta nilai R² Score yang paling tinggi (0.66) dibandingkan model lainnya. Hal ini menunjukkan bahwa Gradient Boosting mampu menjelaskan variasi data target dengan lebih baik.







## Referensi
1. M. Yazdani, "Machine Learning, Deep Learning, and Hedonic Methods for Real Estate Price Prediction," arXiv preprint arXiv:2110.07151, 2021. [Online]. Tersedia: https://arxiv.org/abs/2110.07151
2. O. Pastukh dan V. Khomyshyn, "Using ensemble methods of machine learning to predict real estate prices," arXiv preprint arXiv:2504.04303, 2025. [Online]. Tersedia: https://arxiv.org/abs/2504.04303
3. S. B. Jha, R. F. Babiceanu, V. Pandey, dan R. K. Jha, "Housing Market Prediction Problem using Different Machine Learning Algorithms: A Case Study," arXiv preprint arXiv:2006.10092, 2020. [Online]. Tersedia: https://arxiv.org/abs/2006.10092
_