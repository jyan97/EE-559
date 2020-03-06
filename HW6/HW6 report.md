# EE559 Homework 6 (week 7)

**Jingquan Yan**

**USC ID: 1071912676**

**Email: jingquan@usc.edu**

**EE559 repository:** [**Github**](https://github.com/jyan97/EE-5-5-9)

---

## Problem 1:

### (a):

For this problem, I am using Python and Scikit-Learn.

### (b):

The mean and standard deviation of 13 features in training data are as follows:

<div align=left><img src="C:\git\559\HW6\pic\11.png" style="zoom:65%;" />

<div align=left><img src="C:\git\559\HW6\pic\12.png" style="zoom:70%;" />

|      | 1      | 2     | 3     | 4      | 5      | 6     |
| ---- | ------ | ----- | ----- | ------ | ------ | ----- |
| Mean | 12.965 | 2.27  | 2.376 | 19.649 | 98.91  | 2.272 |
| STD  | 0.82   | 1.103 | 0.273 | 3.465  | 11.559 | 0.615 |

|      | 7     | 8     | 9     | 10    | 11    | 12    | 13      |
| ---- | ----- | ----- | ----- | ----- | ----- | ----- | ------- |
| Mean | 2.029 | 0.361 | 1.576 | 5.091 | 0.953 | 2.56  | 729.708 |
| STD  | 0.92  | 0.12  | 0.541 | 2.404 | 0.23  | 0.723 | 307.221 |

The reason that we have to normalize both the training data and test data with training data factor is that all the information we can derive from the test data (e.g. mean and variance) can be considered as prior knowledge. Our goal is to classify the test data properly with only the training data. If we take advantages of the information of the test data, them the evaluation of the classifier's performance will get biased.

### (c):

1. The Perceptron class is using SGD method to get the optimization result. According to the SGD source code of the base class for SGD, the initial weight vector is supposed to be zero vector.

   <div align=left><img src="C:\git\559\HW6\pic\c1.png" style="zoom:75%;" />

2. The first halting condition is the **tol** argument.  the iterations will stop when (loss > previous_loss - tol). 

   The backup halting condition is the **max_iter** which will stop the iteration when the first condition is not reached and the number of iteration is greater than **max_iter**.

### (d):

1. Classification on training data with normalized training data with first 2 columns:

   <div align=left><img src="C:\git\559\HW6\pic\d1.png" style="zoom:75%;" />

2. Classification on training data with normalized training data with all columns:

   <div align=left><img src="C:\git\559\HW6\pic\d3.png" style="zoom:75%;" />

3. Classification on test data with normalized training data with first 2 columns:

   <div align=left><img src="C:\git\559\HW6\pic\d2.png" style="zoom:75%;" />

4. Classification on test data with normalized training data with all columns:

   <div align=left><img src="C:\git\559\HW6\pic\d4.png" style="zoom:75%;" />

### (e):

1. Classification on training data with normalized training data with first 2 columns:

   <div align=left><img src="C:\git\559\HW6\pic\e1.png" style="zoom:75%;" />

2. Classification on training data with normalized training data with all columns:

   <div align=left><img src="C:\git\559\HW6\pic\e2.png" style="zoom:75%;" />

3. Classification on test data with normalized training data with first 2 columns:

   <div align=left><img src="C:\git\559\HW6\pic\e3.png" style="zoom:75%;" />

4. Classification on test data with normalized training data with all columns:

   <div align=left><img src="C:\git\559\HW6\pic\e4.png" style="zoom:75%;" />

### (f):

1. Comparison between 2 features and 13 features in (d):

   2-features data performs worse than 13-features dataset, both on training data and test data. This makes sense because higher dimension is more likely to get better accuracy than lower feature dimension (given that the dataset are almost linearly separable).

2. Comparison between 2 features and 13 features in (e):

   Similar to 1. 2-features data performs worse than 13-features dataset, both on training data and test data. This makes sense because higher dimension is more likely to get better accuracy than lower feature dimension (given that the dataset are almost linearly separable).

3. Comparison between (d) and (e) on the 2 features:

   Randomly initialize the perceptron weight vector and iterates 100 times performs better than conduct with the default zero weight vector. Because initial weight vector might affect the convergence of the SGD.

4. Comparison between (d) and (e) on the 13 features:

   Similar to 3. Randomly initialize the perceptron weight vector and iterates 100 times performs better than conduct with the default zero weight vector. Because initial weight vector might affect the convergence of the SGD.

### (g):

1. The classification accuracy on test data for first 2 features:

<div align=left><img src="C:\git\559\HW6\pic\g1.png" style="zoom:75%;" />

2. The classification accuracy on test data for 13 features:

<div align=left><img src="C:\git\559\HW6\pic\g2.png" style="zoom:75%;" />

### (h):

1. The classification accuracy on normalized test data for first 2 features:

<div align=left><img src="C:\git\559\HW6\pic\g1.png" style="zoom:75%;" />

2. The classification accuracy on normalized test data for 13 features:

<div align=left><img src="C:\git\559\HW6\pic\g2.png" style="zoom:75%;" />

### (i):

The test-accuracy results of (g) and (h) are identical respectively.

### (j):

1. 2 features in (e) is 0.7865 and in (h) is 0.7528
2. 13 features in (e) is 0.9663 and in (h) is 0.9775

There is not much differences  between perceptron and MSE. 

Comment: In my opinion, Perceptron employs iterations (100 times) to eliminate the effect by the initial weight vector in SGD process and MSE employs offset **b** to get more margin than pure perceptron.  There won't be much random error and the error we can see now might be caused by the undesirable linearly separability of the dataset.

