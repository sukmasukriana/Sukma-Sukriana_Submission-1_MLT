# LAPORAN PROYEK MACHINE LEARNING - Adi Putra Sinaga

> ## Domain Proyek
<p align="justify">
Teknik merupakan bidang yang mengkombinasikan ilmu dan teknologi untuk menyelesaikan suatu permasalahan. Ada banyak sekali ilmu teknik yang bisa kita pelajari, khususnya jika kita melanjutkan studi pada perguruan tinggi yang memiliki konsentrasi pada bidang ini. Beberapa di antaranya teknik fisika, teknik nuklir, teknik metalurgi, teknik perminyakan, teknik nuklir, teknik penerbangan, teknik perkapalan dan masih banyak lagi. Berbekal keilmuan yang telah diperoleh, pada umumnya penguasaan Ilmu Teknik menjadi nilai jual yang patut dibanggakan. Saat ini, lulusan Ilmu Teknik memiliki prospek kerja yang sangat luas. Hampir seluruh sektor membutuhkan ahli teknik. Jenis dan jenjang karier yang ditawarkan cukup bervariasi. Gajinya pun sangat kompetitif dengan melihat kemampuan dan pengalaman yang dimiliki, area tempat bekerja, serta level yang dimasuki. [QuipperCampus](https://campus.quipper.com/majors/id-ilmu-teknik).
</p>

<p align="justify">
Salah satu negara yang memiliki tingkat lulusan pada jurusan teknik tertinggi yaitu India. Negara India memiliki total 6.214 Institusi Teknik dan Teknologi dengan 2,9 juta siswa terdaftar. 1,5 juta siswa setiap tahunnya, mendapatkan gelar di bidang teknik. Kondisi ini tentunya akan memberi bonus demografi bagi negara tersebut. Keterampilan dan kemampuan dibidang teknik tentu harus menyesuaikan dengan pola adaptasi perubahan yang cepat. Namun hanya kurang dari 20 persen lulusan tersebut, memperoleh pekerjaan dibidang teknik disebabkan oleh kurangnya keterampilan yang dibutuhkan untuk melakukan pekerjaan teknis. [BWEDUCATION](http://bweducation.businessworld.in/article/Employability-Of-Engineering-Graduates-In-India-A-Challenge-Needs-To-Address/01-06-2019-171291).
<p>
  
<p align="justify">
Selain itu, faktor upah/gaji/pendapatan memiliki pengaruh terhadap serapan lulusan teknik di India. Apabila terjadi ketidaksesuaian antara upah yang diterima dengan beban kerja yang dilakukan, maka kemungkinan besar bidang ini tidak memiliki daya tarik lebih, sekalipun Bidang teknologi dan Informasi yang akhir-akhir ini sedang berkembang pesat memberi dampak meningkatnya jumlah lapangan pekerjaan baru [timesofindia.indiatimes](https://timesofindia.indiatimes.com/city/kochi/58-of-engg-grads-get-starting-pay-of-only-around-rs-25k-per-mth/articleshow/69779328.cms). Oleh sebab itu peranan pihak penyedia lapangan pekerjaan didalam menetapkan standar upah perlu dilakukan. Berdasarkan uraian diatas, dibutuhkan sebuah strategi management keuangan yang tepat didalam menetapkan upah/gaji/pendapatan bagi lulusan teknik di India. 
</p>
  
> ## Business Understanding
### Problem Statements
<p align="justify">
Bagaimana cara pihak penyedia lapangan pekerjaan menetapkan standar upah/gaji/pendapatan lulusan teknik di India?
Berdasarkan pemaparan sebelumnya, penyedia lapangan pekerjaan membutuhkan sebuah sistem prediksi untuk menetapkan upah/gaji/pendapatan bagi lulusan teknik yang akan dipekerjakan.
</p>

### Goals
<p align="justify">
Berdasarkan uraian pada bagian problem statement, membuat model machine learning regresi dapat menjadi salah satu solusi untuk menentukan upah/gaji/pendapatan berdasarkan pekerjaan yang akan ditawarkan.
</p>


### Solution statements
<p align="justify">
Tujuan proyek ini adalah perusahaan memiliki sebuah sistem prediksi untuk menetapkan upah/gaji/pendapatan bagi lulusan teknik yang akan dipekerjakan berdasarkan data regresi. Model machine learning yang dapat digunakan untuk masalah ini:
</p>

- K-Nearest Neighbor : algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi. Meskipun algoritma KNN mudah dipahami dan digunakan, ia memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar. Permasalahan ini sering disebut sebagai curse of dimensionality (kutukan dimensi). Pada dasarnya, permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data. Jadi, jika kita ingin menggunakan model KNN, perlu dipastikan data yang digunakan memiliki fitur yang relatif sedikit.

- Random Forest : salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni. Namun algoritma ini memiliki kekurangan yaitu: pembelajaran bisa berjalan lambat, tergantung pada parameter yang digunakan dan tidak bisa memperbaiki model yang dihasilkan secara berulang.

- Boosting : Algoritma ini merupakan salah satu dari teknik ensemble untuk menghasilkan beberapa model atau penggolongan untuk prediksi atau klasifikasi, dan juga menggabungkan prediksi dari berbagai model ke dalam prediksi tunggal. Boosting adalah pendekatan iteratif untuk menghasilkan pengklasifikasi yang kuat, yang mampu mencapai kesalahan training seminimal mungkin dari sekelompok pengklasifikasi yang lemah, yang masing-masing hampir tidak dapat melakukan lebih baik daripada tebakan acak. Kelebihan dari algoritma ini adalah mampu meningkatkan akurasi model dan bekerja di sebagian besar kasus. Kekurangan algoritma ini adalah Memakan waktu dan dengan demikian mungkin bukan ide terbaik untuk aplikasi real-time, dan pemilihan model untuk menciptakan ensemble adalah seni yang benar-benar sulit untuk dikuasai. 

> ## Data Understanding
Dataset diperoleh dari [Kaggle](https://www.kaggle.com/). ***Kaggle*** merupakan platform penyedia dataset untuk data science. Untuk proyek ini, dataset yang saya pakai yaitu: 
- [Dataset](https://www.kaggle.com/manishkc06/engineering-graduate-salary-prediction)
  > Dataset yang kita gunakan merupakan dataset yang berisi informasi terkait upah/gaji/pendapatan anak lulusan teknik di India. Dataset ini memberikan memaparkan data terkait berbagai faktor seperti nilai perguruan tinggi, keterampilan kandidat, kedekatan perguruan tinggi dengan pusat industri, spesialisasi yang dimiliki, kondisi pasar untuk industri tertentu menentukan upah/gaji/pendapatan anak lulusan teknik di India. 

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

Berikut informasi mengenai jumlah data ,tipe data dan informasi data hilang (***missing value***) yang terdapat pada dataset ini:
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/df-info.png?raw=true"/>
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/df-null-check.png?raw=true"/>
</p>
 
Dalam memudahkan proses analisis diperlukan beberapa visualisasi data, seperti:  
- ***sns.boxplot***, untuk mendeteksi adanya data yang berada di luar batas atas dan batas bawah data (***outliers***).
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/collegeGPA-rev.png?raw=true"/>
</p>

> Kita dapat melihat, terdapat titik-titik yang berada di luar range data fitur utama yang menandakan adanya outliers.


- ***count.plot***, untuk menganalisa fitur specialization.
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/specialization-rev.png?raw=true"/>
</p>

> Kita dapat melihat, pada fitur specialization , ***computer enggineering***, ***electronics and communication enggineering*** dan ***information technology*** menjadi bidang dengan serapan tertinggi pada lulusan teknik India didunia pekerjaan.

Penerapan visualisasi data pada gambar diatas, dapat menunjukkan banyaknya jumlah sampel dan persentase pada fitur kategori Specialization.

- ***sns.catplot***, untuk mempertimbangkan Fitur Salary dengan fitur kategori, 
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/multivariate-analysis-rev.png?raw=true"/>
</p>

> Beberapa hal yang dapat kita peroleh dari visualisai informasi tersebut adalah :
  -
    - Upah antara pekerja wanita(f) dan pria(m) relatif sama.
    - Upah yang diperoleh berdasarkan fitur specialization adalah kisaran 15000 - 35000 INR(***Indian Rupee***)
    - Upah yang diperoleh berdasarkan tempat lulusan teknik berasal relatif sama, namun bila ditetapkan informasi kisaran upah, kisarannya dimulai dari 20000 - 40000 INR(***Indian Rupee***)

<hr>

- ***sns.pairplot***, untuk menunjukkan semua grafik fitur numerik,
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/plot%20pair%20fitur.png?raw=true"/>
</p>

> **NOTE:** fitur salary terdapat pada bagian paling bawah. Kita belum dapat menarik kesimpulan, dikarenakan sebaran data yang masih acak(***random***)

- ***sns.heatmap***, untuk menunjukkan matrik korelasi fitur numerik.
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/matrik%20korelasi%20fitur%20numerik-a-rev.png?raw=true"/>
</p>

> Pada visualisasi informasi diatas, dapat disimpulkan bahwa:
- 
  - fitur *extraversion* memiliki nilai matrik korelasi cenderung negatif. Namun dapat disimpulkan juga bahwa fitur tersebut juga memiliki matrik korelasi paling lemah dari antara fitur lain yaitu -0.03.
  - fitur *Quant* memiliki nilai matrik korelasi cenderung positif paling tinggi dari antara fitur lain yaitu 0.3.
  - Nilai kisaran matrik korelasi dari semua fitur yaitu -0.03 - 0.3. 

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
 
- Melakukan pemeriksaan terhadap nilai yang hilang(missing value) pada dataset 

<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/null-check.png?raw=true"/>
</p>

- Memeriksa outlier dengan metode IQR.
  - ***Outliers*** adalah sampel yang nilainya sangat jauh dari cakupan umum data utama. Ia adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya. Kita dapat menangani outliers dengan teknik IQR method.
  - ***IQR*** adalah singkatan dari Inter Quartile Range. Untuk memahami apa itu IQR, dibutuhkan pemahaman terhadap konsep kuartil. Kuartil dari suatu populasi adalah tiga nilai yang membagi distribusi data menjadi empat sebaran. Seperempat dari data berada di bawah kuartil pertama (Q1), setengah dari data berada di bawah kuartil kedua (Q2), dan tiga perempat dari data berada di kuartil ketiga (Q3). Dengan demikian interquartile range atau IQR = Q3 - Q1. Kita dapat menggunakan metode IQR untuk mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier.

<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/Logical-rev.png?raw=true"/>
</p>

> **NOTE:** Pada plot box, memperlihatkan beberapa titik yang berada di luar range data fitur utama. Untuk mengatasi outlier ini, metode IQR  digunakan dalam analisis statistik untuk membantu menarik kesimpulan mengenai sekumpulan data. 

Untuk data preparation, beberapa teknik yang diperlukan yaitu :
- **One Hot Encoding:** metode ini dilakukan pada fitur kategori karena model machine learning lebih mudah memehami data apabila berupa angka atau biner
<p align="center">
  <img src="https://github.com/adiputrasinaga-cmd/Predictive-Analytics/blob/main/img/one-hot-encoding.png?raw=true"/>
</p>

- **Dimensionality Reduction :** Mengurangi fitur atau kolom yang banyak menjadi lebih sedikit dengan tetap mempertahankan informasi pada data tersebut. 

- ***Train Test Split*** : Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. 

- ***Data Transform***

  - ***Scaling:***
      >   - ***Scaling Data Test*** : kita perlu melakukan proses scaling fitur numerik pada data test/uji. Hal ini harus dilakukan agar skala antara data train dan data test sama dan kita bisa melakukan evaluasi.
      >   - ***Standarisasi*** : Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses standarisasi dapat membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.   

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

  >- KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika kita memilih k yang terlalu rendah, maka akan menghasilkan model yang overfit dan hasil prediksinya memiliki varians tinggi. Jika kita memilih k terlalu tinggi, maka model yang dihasilkan akan underfit dan prediksinya memiliki bias yang tinggi. Kita dapat mencoba beberapa nilai k yang berbeda, kemudian membandingkan mana nilai yang paling sesuai untuk model. Pada kasus ini, kita akan menetapkan nilai k(n_neighbors) = 100.

- Model prediksi dengan algoritma Random Forest:

```
RF = RandomForestRegressor(n_estimators=100, max_depth=1, random_state=123, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train) 
```
   > Berikut merupakan penjelasan terhadap setiap parameter yang digunakan: 
      - **n_estimators** = menunjukkan jumlah model Decision Tree yang digunakan pada Random Forest
      - **max_depth** = menunjukkan kedalaman maksimum Decision Tree
      - **random_state** = mengontrol random number generator yang digunakan. Parameter ini berupa bilangan integer dan nilainya bebas. Parameter ini bertujuan untuk  memastikan bahwa hasil pembagian dataset konsisten dan memberikan data yang sama setiap kali model dijalankan. Jika tidak  ditentukan, maka tiap kali melakukan split, kita akan mendapatkan data train dan tes berbeda. Hal ini berpengaruh terhadap akurasi model ML yang menjadi berbeda tiap kali di-run.
      - **n_jobs = -1** = jumlah task/pekerjaan yang harus dijalankan secara paralel untuk kecocokan dan prediksi.     

- Model prediksi dengan algoritma Boosting Algorithm - Adaptive Boosting:

```
boosting = AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=256)
boosting.fit(X_train, y_train)
models.loc['train_mse', 'Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train),y_true=y_train)
```
  > Parameter yang akan kita bahas yaitu Learning rate. Learning rate merupakan salah satu parameter training untuk menghitung nilai koreksi bobot pada waktu proses training. Nilai learning rate ini berada pada range nol (0) sampai (1). Semakin besar nilai learning rate, maka proses training akan berjalan semakin cepat. Semakin besar learning rate, maka ketelitian jaringan akan semakin berkurang, tetapi berlaku sebaliknya, apabila learning rate-nya semakin kecil, maka ketelitian jaringan akan semakin besar atau bertambah dengan konsekuensi proses training akan memakan waktu yang semakin lama. Pada kasus ini kita memilih nilai learning_rate = 0.05. 
  
  >**NOTE:** Setiap nilai parameter yang kita gunakan masih bersifat percobaan( ***experimental***), sehingga diharapkan bagi kita untuk dapat mengeksplore lebih jauh lagi terhadap setiap nilai yang telah kita gunakan. 
 
 Untuk hasil dari setiap model yang telah kita buat, berikut akan dijelaskan pada tahap Evaluasi.

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

Untuk mengingatkan kembali, kita harus memastikan bahwa tahapan scaling data test/uji sudah dilakukan seperti penjelasan pada tahapan ***Data Preparation***.
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

> Dari gambar di atas, terlihat bahwa model KNN memberikan nilai eror yang paling kecil pada data test/uji dibandingkan model lain. 

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

Kita dapat melihat bahwa model dengan algoritma KNN memiliki nilai prediksi (KNN = 308050)  lebih dekat dengan nilai sebenarnya (nilai sebenarnya = 315000) dibandingkan model dengan algoritma random forest (RF = 280280.9) dan boosting (Boosting = 293776.4).

> ## Kesimpulan
<p align="justify">
Pengujian setiap model dengan algoritma yang berbeda menghasilkan nilai prediksi yang berbeda pula. Model dengan nilai yang mendekati nilai sebenarnya diperoleh pada prediksi dengan menggunakan algoritma K-Nearest Neighbor. Untuk prediksi menggunakan algoritma Random Forest dan Boosting, performanya masih dibawah prediksi model KNN. Sehingga dapat disimpulkan bahwa pada kasus ini, model dengan menggunakan Algoritma KNN lebih tepat untuk digunakan atau diterapkan.
</p>

> ## Penutup
<p align="justify">
Sekian hasil dari laporan proyek machine learning, predicitive analytics ini. Bilamana didalam penyampaian serta penjelasan yang kurang berkenaan, saya memohon maaf. Atas waktu dan perhatiannya, saya ucapkan Terima kasih telah membaca laporan ini. Semoga dapat memberi manfaat bagi kita semuanya.
</p>

  
