Nama : Ira Dwita Syafitri Tarigan 

NPM : 1184024

Kelas : D4 TI 3A



Linear Support Vector Machine (SVM)
==

![](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
[![PyPI](https://img.shields.io/pypi/pyversions/Django.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1045859.svg)](https://doi.org/10.5281/zenodo.1045859)

Support vector machine (SVM) dikembangkan oleh Vapnik, dan telah digunakan di banyak aplikasi dunia nyata, terutama dalam kasus klasifikasi biner.
Tujuan utamanya adalah menemukan hyperplane optimal yang memisahkan dua kelas dalam data _D_ tertentu. Klasifikasi data dilakukan dengan menggunakan fungsi keputusan _f (x) _:

![](assets/input.png)

![](assets/decision_function.png)

di mana `{-1, + 1}` adalah kelas dari data yang diberikan. Parameter pembelajaran (bobot `w`, dan bias` b`) diperoleh sebagai solusi dari masalah pengoptimalan berikut:

![](assets/constrained-svm.png)

![](assets/euclidean-norm.png)

![](assets/constraint-1.png)

![](assets/constraint-2.png)

di mana `|| w || _ {2}` adalah norma Euclidean (juga dikenal sebagai norma L2), `\ xi` adalah fungsi biaya, dan` C` adalah parameter penalti (yang mungkin berupa nilai arbitrer atau nilai yang diperoleh melalui penyetelan hyper-parameter). Masalah pengoptimalan tak terbatas yang sesuai adalah sebagai berikut:

![](assets/l1-svm.png)

di mana `wx + b` adalah fungsi yang mengembalikan vektor yang berisi skor untuk setiap kelas (yaitu kelas yang diprediksi). Sasaran persamaan di atas dikenal sebagai bentuk primal L1-SVM, dengan kerugian engsel standar. Dalam proyek ini, varian L2-SVM dari SVM digunakan karena dapat dibedakan, dan memberikan hasil yang lebih stabil daripada L1-SVM.

![](assets/l2-svm.png)

Untuk implementasi ini, SVM ditulis menggunakan Python dan TensorFlow (sebagai pustaka kecerdasan mesin), dan masalah yang ditangani adalah klasifikasi biner menggunakan [Wisconsin diagnostic dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).


Berikut adalah distribusi kelas dari dataset tersebut:

| Kelas | Jumlah kejadian |
| ----- | ------------------- |
| jinak | 357 |
| Ganas | 212 |

Sebanyak 569 instans. Dalam implementasi ini, kelasnya adalah `{-1, +1}`, masing-masing mewakili kelas jinak dan kelas ganas.

Fitur-fitur yang termasuk dalam kumpulan data tersebut adalah sebagai berikut:

* radius
* tekstur
* perimeter
* area
* kehalusan
* kekompakan
* cekung
* titik cekung
* simetri
* dimensi fraktal

Setiap fitur memiliki (1) mean, (2) kesalahan standar, dan (3) "terburuk" atau terbesar (rata-rata dari tiga nilai terbesar) yang dihitung. Oleh karena itu, dataset memiliki 30 fitur.

| Variabel | Contoh | Bentuk |
| -------- | --------- | ----- |
| x (fitur) | 569 | (569, 30) |
| y (label) | 569 | (569) |

## Prasyarat

Direkomendasikan agar Anda menginstal Python 3.x (khususnya 3.5 atau 3.6) di sistem Anda. Instal pustaka Python yang ditentukan dalam perintah berikut untuk menjalankan program.


First, clone the project.
```
~$ git clone https://github.com/afagarap/support-vector-machine.git/
```

Program parameters.

```buildoutcfg
usage: main.py [-h] -c SVM_C -n NUM_EPOCHS -l LOG_PATH

SVM built using TensorFlow, for Wisconsin Breast Cancer Diagnostic Dataset

optional arguments:
  -h, --help            show this help message and exit

Arguments:
  -c SVM_C, --svm_c SVM_C
                        Penalty parameter C of the SVM
  -n NUM_EPOCHS, --num_epochs NUM_EPOCHS
                        number of epochs
  -l LOG_PATH, --log_path LOG_PATH
                        path where to save the TensorBoard logs
```

Then, go to its directory by using `cd`, and run the main program according to your desired parameters.
```
~$ cd support-vector-machine
~/support-vector-machine$ python3 main.py --svm_c 1 --num_epochs 1000 --log_path ./logs
```

## Sample Result

The hyper-parameters used in the experiment were assigned by hand, and not through optimization/tuning.

#### Hyper-parameters used for the SVM
|Hyperparameters|SVM|
|--------------|------|
|BATCH_SIZE|4
|EPOCHS|1000|
|LEARNING RATE|1e-3|
|SVM_C|1|


Training accuracy (graph above), and training loss (graph below).

![](assets/loss_and_accuracy.png)

Truncated training loss and training accuracy, with counts of true negative, false negative, true positive, and false positive.

```
step[0] train -- loss : 1310.61669921875, accuracy : 0.32500001788139343
step[100] train -- loss : 754.3006591796875, accuracy : 0.32500001788139343
step[200] train -- loss : 580.3919677734375, accuracy : 0.3499999940395355
...
step[10800] train -- loss : 5.456733226776123, accuracy : 1.0
step[10900] train -- loss : 6.086201190948486, accuracy : 0.9749999642372131
EOF -- training done at step 10999
Validation accuracy : 0.949999988079071
True negative : 12
False negative : 2
True positive : 26
False positive : 0
```

Confusion matrix on test data.

![](assets/confusion_matrix.png)


#### Standardized Dataset
The results above are based on a raw dataset from `sklearn`, i.e. `sklearn.datasets.load_breast_cancer().data`. Now, the following is a sample output based on a standardized dataset (using `sklearn.preprocessing.StandardScaler`):

![](assets/loss_and_accuracy_based_on_standardized_data.png)

Truncated training loss and training accuracy, with counts of true negative, false negative, true positive, and false positive.

```buildoutcfg
step[0] train -- loss : 86.02317810058594, accuracy : 0.44999998807907104
step[100] train -- loss : 49.41931915283203, accuracy : 0.6250000596046448
step[200] train -- loss : 41.406898498535156, accuracy : 0.925000011920929
...
step[10800] train -- loss : 2.045114040374756, accuracy : 1.0
step[10900] train -- loss : 6.896279335021973, accuracy : 0.9749999642372131
EOF -- training done at step 10999
Validation accuracy : 0.9750000238418579
True negative : 16
False negative : 0
True positive : 23
False positive : 1
```

Confusion matrix on the standardized test data.

![](assets/confusion_matrix_based_on_standardized_data.png)

## Citation

To cite the repository/software, kindly use the following BibTex entry:
```
@misc{abien_fred_agarap_2017_1045859,
  author       = {Abien Fred Agarap},
  title        = {AFAgarap/support-vector-machine v0.1.5-alpha},
  month        = nov,
  year         = 2017,
  doi          = {10.5281/zenodo.1045859},
  url          = {https://doi.org/10.5281/zenodo.1045859}
}
```

## License

```buildoutcfg
Copyright 2017 Abien Fred Agarap

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
