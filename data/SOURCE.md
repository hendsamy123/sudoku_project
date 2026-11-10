# Dataset source: MNIST (digits 1-9 used)

* Official page: http://yann.lecun.com/exdb/mnist/  (Y. LeCun, C. Cortes, C. J. C. Burges)
* What was actually downloaded: the 4 original gzip files, fetched from the GitHub mirror
  https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/
  because the official host (yann.lecun.com) and the S3 mirror were not reachable from the build environment.
* Integrity: MD5 of all 4 files equals the widely published MNIST checksums (as used by torchvision) and the IDX headers/counts match (60,000 train, 10,000 test, 28x28).

| File | Bytes | MD5 |
|---|---|---|
| train-images-idx3-ubyte.gz | 9,912,422 | f68b3c2dcbeaaa9fbdd348bbdeb94873 |
| train-labels-idx1-ubyte.gz | 28,881 | d53e105ee54ea40749a09fcbcd1e9432 |
| t10k-images-idx3-ubyte.gz | 1,648,877 | 9fb629c4189551a2d022fa330f9573f3 |
| t10k-labels-idx1-ubyte.gz | 4,542 | ec29112dd5afa0611ce80d1b7f02629c |

`mnist_1to9.npz` = processed file made by `python -m data.make_data` (seed 7): digit 0 removed; 44,077 train / 10,000 validation
(random part of the original training set) / 9,020 test (original test set, untouched).
Arrays: x_train, y_train, x_val, y_val, x_test, y_test (images uint8 28x28, labels 1..9).
Licence: MNIST is freely available for research/education (no registration); cite LeCun et al. 1998. Check the licence line on the official page before redistributing.
