# LAPORAN PROYEK MACHINE LEARNING - Adi Putra Sinaga

> ## Domain Proyek
<p align="justify">
Teknik merupakan bidang yang mengkombinasikan ilmu dan teknologi untuk menyelesaikan suatu permasalahan. Ada banyak sekali ilmu teknik yang bisa kita pelajari, khususnya jika kita melanjutkan studi pada perguruan tinggi yang memiliki konsentrasi pada bidang ini. Beberapa di antaranya teknik fisika, teknik nuklir, teknik metalurgi, teknik perminyakan, teknik nuklir, teknik penerbangan, teknik perkapalan dan masih banyak lagi. Berbekal keilmuan yang telah diperoleh, pada umumnya penguasaan Ilmu Teknik menjadi nilai jual yang patut dibanggakan. Saat ini, lulusan Ilmu Teknik memiliki prospek kerja yang sangat luas,. Hampir seluruh sektor membutuhkan ahli teknik. Jenis dan jenjang karier yang ditawarkan cukup bervariasi. Gajinya pun sangat kompetitif dengan melihat kemampuan dan pengalaman yang dimiliki, area tempat bekerja, serta level yang dimasuki. [QuipperCampus](https://campus.quipper.com/majors/id-ilmu-teknik).
</p>

<p align="justify">
Salah satu negara yang memiliki tingkat lulusan pada jurusan teknik tertinggi yaitu India. Negara India memiliki total 6.214 Institusi Teknik dan Teknologi dengan 2,9 juta siswa terdaftar. 1,5 juta siswa setiap tahunnya, mendapatkan gelar di bidang teknik. Kondisi ini tentunya akan memberi bonus demografi bagi negara tersebut. Keterampilan dan kemampuan dibidang teknik tentu harus menyesuaikan dengan pola adaptasi perubahan yang cepat. Namun hanya kurang dari 20 persen lulusan tersebut, memperoleh pekerjaan dibidang teknik disebabkan oleh kurangnya keterampilan yang dibutuhkan untuk melakukan pekerjaan teknis. [BWEDUCATION](http://bweducation.businessworld.in/article/Employability-Of-Engineering-Graduates-In-India-A-Challenge-Needs-To-Address/01-06-2019-171291). Berdasarkan uraian diatas, tentu perusahaan di India memerlukan sebuah strategi management keuangan yang tepat didalam memberikan upah/gaji/pendapatan bagi lulusan teknik di India. 
<p>
  
<p align="justify"></p>
  
> ## Business Understanding
### Problem Statements
<p align="justify">
Berdasarkan pemaparan sebelumnya, perusahaan membutuhkan sebuah sistem prediksi untuk menetapkan upah/gaji/pendapatan bagi lulusan teknik yang akan dipekerjakan.
</p>

### Goals
<p align="justify">
Membuat model machine learning regresi, untuk menentukan upah/gaji/pendapatan berdasarkan pekerjaan yang akan ditawarkan.
</p>


### Solution statements
<p align="justify">
Tujuan proyek ini adalah perusahaan memiliki sebuah sistem prediksi untuk menetapkan upah/gaji/pendapatan bagi lulusan teknik yang akan dipekerjakan berdasarkan data regresi. Model machine learning yang dapat digunakan untuk masalah ini:
</p>

- K-Nearest Neighbor : algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi. Pada modul ini, kita akan menggunakannya untuk kasus regresi.

- Random Forest : salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni. 

- Boosting : Algoritma ini merupakan salah satu dari teknik ensemble untuk menghasilkan beberapa model atau penggolongan untuk prediksi atau klasifikasi, dan juga menggabungkan prediksi dari berbagai model ke dalam prediksi tunggal. Boosting adalah pendekatan iteratif untuk menghasilkan pengklasifikasi yang kuat, yang mampu mencapai kesalahan training seminimal mungkin dari sekelompok pengklasifikasi yang lemah, yang masing-masing hampir tidak dapat melakukan lebih baik daripada tebakan acak. Kelebihan dari algoritma ini adalah mampu meningkatkan akurasi model dan bekerja di sebagian besar kasus. Kekurangan algoritma ini adalah Memakan waktu dan dengan demikian mungkin bukan ide terbaik untuk aplikasi real-time, dan pemilihan model untuk menciptakan ensemble adalah seni yang benar-benar sulit untuk dikuasai. 

> ## Data Understanding
Dataset diperoleh dari [Kaggle](https://www.kaggle.com/). Untuk proyek ini, dataset yang saya pakai yaitu: 
- https://www.kaggle.com/manishkc06/engineering-graduate-salary-prediction

Berikut penjelasan mengenai variabel-variabel pada kolom dataset:
| Variabel | Deskripsi |
| --- | --- |
| ID | Nilai unik untuk mengidentifikasi kandidat |
| Salary | CTC tahunan ditawarkan kepada kandidat (dalam INR) |
| Gender | Jenis kelamin kandidat |
| DOB | Tanggal lahir kandidat |
| 10percentage | Nilai keseluruhan diperoleh dalam ujian kelas 10 |
| 10board | Dewan sekolah yang kurikulumnya diikuti kandidat di kelas 10 |
| 12graduation | Tahun kelulusan - sekolah menengah atas |
| 12percentage | Nilai keseluruhan diperoleh dalam ujian kelas 12 |
| 12board | Dewan sekolah yang kurikulumnya diikuti kandidat |
| CollegeID |  ID unik yang mengidentifikasi universitas/perguruan tinggi tempat kandidat menghadiri untuk sarjananya |
| CollegeTier | Setiap perguruan tinggi telah dianotasi sebagai 1 atau 2. Anotasi telah dihitung dari rata-rata skor yang diperoleh siswa di perguruan tinggi/universitas. Perguruan tinggi dengan skor rata-rata di atas ambang batas ditandai sebagai 1 dan lainnya sebagai 2 |
| Degree| Gelar yang diperoleh / dikejar oleh kandidat |
| Specialization| Spesialisasi yang dikejar oleh kandidat |
| CollegeGPA| IPK Agregat saat kelulusan |
| CollegeCityID| ID unik untuk mengidentifikasi kota tempat perguruan tinggi berada |
| CollegeCityTier| Tingkat kota tempat perguruan tinggi berada. Ini dianotasi berdasarkan populasi kota |
| CollegeState| Nama negara bagian di mana perguruan tinggi berada |
| GraduationYear| Tahun kelulusan (gelar Sarjana)|
| English | Skor di bagian Bahasa Inggris |
| Logical | Skor di bagian Kemampuan logis |
| Quant | Skor di bagian kemampuan Kuantitatif |
| Domain | Skor di bagian kemampuan Kuantitatif |
| ComputerProgramming | Skor di bagian Pemrograman Komputer|
| ElectronicsAndSemicon | Skor di bagian Teknik Elektronik & Semikonduktor |
| ComputerScience | Skor di bagian Ilmu Komputer |
| MechanicalEngg | Skor di bagian Teknik Mesin |
| ElectricalEngg | Skor di bagian Teknik Elektro |
| TelecomEngg|Skor di bagian Teknik Telekomunikasi |
| CivilEngg| Skor di bagian Teknik Sipil |
| conscientiousness | Skor di salah satu bagian dari tes kepribadian |
| agreeableness | Skor di salah satu bagian dari tes kepribadian |
| extraversion | Skor di salah satu bagian dari tes kepribadian |
| nueroticism | Skor di salah satu bagian dari tes kepribadian |
| openesstoexperience | Skor di salah satu bagian dari tes kepribadian |
 
Dalam memudahkan proses analisis diperlukan beberapa visualisasi data, seperti:  
- ***sns.boxplot***, untuk mendeteksi adanya data yang berada di luar batas atas dan batas bawah data (***outliers***).
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/collegeGPA-rev.png?raw=true"/>
</p>

- ***count.plot***, untuk menganalisa fitur.
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/specialization-rev.png?raw=true"/>
</p>

Penerapan visualisasi data pada gambar diatas, dapat menunjukkan banyaknya jumlah sampel dan persentase pada fitur kategori Specialization.

- ***sns.catplot***, untuk mempertimbangkan Fitur Salary dengan fitur kategorikal, 
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/multivariate-analysis-rev.png?raw=true"/>
</p>


- ***sns.pairplot***, untuk menunjukkan semua grafik fitur numerik,
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/plot%20pair%20fitur.png?raw=true"/>
</p>

- ***sns.heatmap***, untuk menunjukkan matrik korelasi fitur numerik.
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/matrik%20korelasi%20fitur%20numerik-a-rev.png?raw=true"/>
</p>

Dengan visualisasi data yang telah dilakukan, diharapkan dapat memudahkan kita didalam proses analisa data.

> ## Data Preparation 
Beberapa langkah yang perlu kita dilakukan sebelum melakukan tahapan Data Preparation:

- Menghilangkan fitur yang tidak diperlukan : untuk meminimalisasi fitur yang tidak memiliki pengaruh signifikan dengan tujuan proyek ini, seperti: 

| Nama kolom | Deskripsi |
| --- | --- |
| ID | Nilai unik untuk mengidentifikasi kandidat |
| DOB | Tanggal lahir kandidat |
| 10percentage | Nilai keseluruhan diperoleh dalam ujian kelas 10 |
| 10board | Dewan sekolah yang kurikulumnya diikuti kandidat di kelas 10 |
| 12graduation | Tahun kelulusan - sekolah menengah atas |
| 12percentage | Nilai keseluruhan diperoleh dalam ujian kelas 12 |
| 12board | Dewan sekolah yang kurikulumnya diikuti kandidat |
| CollegeID |  ID unik yang mengidentifikasi universitas/perguruan tinggi tempat kandidat menghadiri untuk sarjananya |
| CollegeTier | Setiap perguruan tinggi telah dianotasi sebagai 1 atau 2. Anotasi telah dihitung dari rata-rata skor yang diperoleh siswa di perguruan tinggi/universitas. Perguruan tinggi dengan skor rata-rata di atas ambang batas ditandai sebagai 1 dan lainnya sebagai 2 |
| Degree| Gelar yang diperoleh / dikejar oleh kandidat |
| CollegeCityID| ID unik untuk mengidentifikasi kota tempat perguruan tinggi berada |
| CollegeCityTier| Tingkat kota tempat perguruan tinggi berada. Ini dianotasi berdasarkan populasi kota |
| GraduationYear| Tahun kelulusan (gelar Sarjana)|
 
Untuk menghilangkan kolom/fitur tersebut dapat dilakukan dengan cara:
```
df.drop(
    ['ID', 'DOB', '10board', '12board', 'CollegeID', 'CollegeTier', 
     '10percentage','12graduation', '12percentage', 'CollegeCityID', 
     'CollegeCityTier','Degree','GraduationYear'], 
    axis='columns', 
    inplace=True
)
```
> **NOTE:** variabel ***df*** merupakan dataframe.


- Melakukan pemeriksaan terhadap nilai yang hilang(missing value) pada dataset 
```
df.isnull().sum()
```
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/null-check.png?raw=true"/>
</p>

- Memeriksa outlier dengan metode IQR.
  - ***Outliers*** adalah sampel yang nilainya sangat jauh dari cakupan umum data utama. Ia adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya. Kita dapat menangani outliers dengan teknik IQR method.
  - ***IQR*** adalah singkatan dari Inter Quartile Range. Untuk memahami apa itu IQR, dibutuhkan pemahaman terhadap konsep kuartil. Kuartil dari suatu populasi adalah tiga nilai yang membagi distribusi data menjadi empat sebaran. Seperempat dari data berada di bawah kuartil pertama (Q1), setengah dari data berada di bawah kuartil kedua (Q2), dan tiga perempat dari data berada di kuartil ketiga (Q3). Dengan demikian interquartile range atau IQR = Q3 - Q1. Kita dapat menggunakan metode IQR untuk mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier.

```    
sns.boxplot(x=df['Logical'])
```

<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/Logical-rev.png?raw=true"/>
</p>

> **NOTE:** Perintah tersebut menampilkan sebuah plot box yang memperlihatkan beberapa titik yang berada di luar range data fitur utama. Untuk mengatasi outlier ini, metode IQR  digunakan dalam analisis statistik untuk membantu menarik kesimpulan mengenai sekumpulan data. Untuk menggunakan metode IQR, dapat menggunakan perintah:

```
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3-Q1
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]
```


Untuk data preparation, beberapa teknik yang diperlukan yaitu :
- **One Hot Encoding:** metode ini dilakukan pada fitur kategori karena model machine learning lebih mudah memehami data apabila berupa angka atau biner
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/one-hot-encoding.png?raw=true"/>
</p>

- **Reduksi dimensi :** Mengurangi fitur atau kolom yang banyak menjadi lebih sedikit dengan tetap mempertahankan informasi pada data tersebut. 
```
pca = PCA(n_components=1, random_state=123)

pca.fit(
    df[['English','Logical','Quant','Domain','ComputerProgramming',
        'ElectronicsAndSemicon','conscientiousness','agreeableness',
        'extraversion','nueroticism','openess_to_experience']]
)

df['AMCATscore'] = pca.transform(
    df.loc[:, ('English','Logical','Quant','Domain','ComputerProgramming',
              'ElectronicsAndSemicon','conscientiousness','agreeableness',
              'extraversion','nueroticism','openess_to_experience')]
).flatten()

df.drop(
    ['English','Logical','Quant','Domain','ComputerProgramming',
      'ElectronicsAndSemicon','conscientiousness','agreeableness',
      'extraversion','nueroticism','openess_to_experience'],
      axis=1, inplace=True
) 
```

Pada gambar dibawah ini, terjadi proses reduksi yang semula 11 fitur yang berkaitan menjadi 1 fitur bernama AMCATscore. 
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/reduct-dim.png?raw=true"/>
</p>

- ***Train Test Split*** : Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. 
```
X = df.drop(["Salary"], axis=1)
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state=123)
```
> **NOTE:** Segala variabel yang digunakan pada contoh kode disesuaikan terhadap dataframe pada dataset yang digunakan

- ***Standarisasi*** : Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses standarisasi dapat membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. 
```
numeric_feature = ['collegeGPA', 'AMCATscore']
scaler = StandardScaler()
scaler.fit(X_train[numeric_feature])
X_train[numeric_feature] = scaler.transform(X_train.loc[:, numeric_feature])
X_train[numeric_feature]
```


> ## Modeling
> <p align="justify">
Terdapat beberapa algoritma yang dapat diterapkan pada kasus regresi. Mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik adalah cara yang dapat kita lakukan sebagai solusi utama. Ketiga algoritma yang digunakan, antara lain: 
</p>
<ol type="1">
    <li>K-Nearest Neighbor (KNN)</li>
    <li>Random Forest</li>
    <li>Boosting Algorithm:
      <ul>
        <li>Adaptive Boosting</li>
      </ul>
    </li>
</ol>

- Sebelum membuat ketiga model diatas, terlebih dahulu kita mempersiapkan dataframe untuk analisa model:

```
models = pd.DataFrame(
    index=['train_mse', 'test_mse'],
    columns=['KNN', 'RandomForest', 'Boosting']
)
```

- Model prediksi dengan algoritma KNN:

```
KNN = KNeighborsRegressor(n_neighbors=100)
KNN.fit(X_train, y_train)
y_pred_KNN = KNN.predict(X_train)
```
- Model prediksi dengan algoritma Random Forest:

```
RF = RandomForestRegressor(n_estimators=100, max_depth=1, random_state=123, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train) 
```

- Model prediksi dengan algoritma Boosting Algorithm - Adaptive Boosting:

```
boosting = AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=256)
boosting.fit(X_train, y_train)
models.loc['train_mse', 'Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train),
                                                         y_true=y_train)
```

>## Evaluation
<p align="justify">
Mengevaluasi model regresi sebenarnya relatif sederhana. Secara umum, hampir semua metrik adalah sama. Jika prediksi mendekati nilai sebenarnya, performanya baik. Sedangkan jika tidak, performanya buruk. Secara teknis, selisih antara nilai sebenarnya dan nilai prediksi disebut eror. Maka, semua metrik mengukur seberapa kecil nilai eror tersebut.
</p>

<p align="justify">
Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut:
</p>

<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/mse.png?raw=true"/>
</p>

Sebelum menghitung nilai MSE dalam model, kita perlu melakukan proses scaling fitur numerik pada data test. Kita perlu melakukan scaling terhadap data uji. Hal ini harus dilakukan agar skala antara data train dan data test sama dan kita bisa melakukan evaluasi.
Untuk proses scaling, dapat disesuaikan seperti contoh kode program: 

```
X_test.loc[:, numerik_fitur] = scaler.transform(X_test[numerik_fitur])
```

Setelah itu ketiga model bisa di evaluasi dengan metrik MSE. Penggunaan metrik MSE dapat disesuaikan seperti contoh kode program:

```
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN', 'RF', 'Boosting'])
model_dict = {'KNN': KNN, 'RF': RF, 'Boosting':boosting}

for name, model in model_dict.items():
  mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e6
  mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e6
```

<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/mse-model.png?raw=true"/>
</p>

- Membuat Plot metrik MSE dengan bar chart:
```
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
```
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/plot%20metrik%20mse-rev.png?raw=true"/>
</p>

- Pengujian model prediksi menggunakan nilai Salary dari dataset:

```
prediksi = X_test.iloc[:1].copy()
pred_dict = {'Salary':y_test[:1]}

for name, model in model_dict.items():
  pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
```
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/predict-result.png?raw=true"/>
</p>

<p align="justify">
Pengujian setiap model dengan algoritma yang berbeda menghasilkan nilai prediksi yang berbeda pula. Model dengan nilai yang mendekati nilai sebenarnya diperoleh pada prediksi dengan menggunakan algoritma K-Nearest Neighbor. Untuk prediksi menggunakan algoritma Random Forest dan Boosting, performanya masih dibawah prediksi model KNN. Sehingga dapat disimpulkan bahwa pada kasus ini, model dengan menggunakan Algoritma KNN lebih tepat untuk digunakan atau diterapkan.
</p>


> ## Penutup
Sekian hasil dari laporan proyek machine learning, predicitive analytics ini. Bilamana didalam penyampaian serta penjelasan yang kurang berkenaan, saya memohon maaf. Atas waktu dan perhatiannya, saya ucapkan Terima kasih telah membaca laporan ini. Semoga dapat memberi manfaat bagi kita semuanya.

  
