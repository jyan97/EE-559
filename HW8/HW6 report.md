# EE559 Homework 8

**Jingquan Yan**

**USC ID: 1071912676**

**Email: jingquan@usc.edu**

**EE559 repository:** [**Github**](https://github.com/jyan97/EE-5-5-9)

---

## Problem 1:

### (a):

The slack parameter **C** is introduced to trade-off between the misclassified penalty and the soft-margin. For non-linearly separable dataset, as we increase the value of **C**,  the penalty for misclassified data increases and in other words, the soft-margin gets relatively smaller(rigorous) and vice versa.

We plug in **C** as 1, 5, 10, 50, 100 and the respective accuracies are:

| **C** | Accuracy |
| ----- | -------- |
| 1     | 0.9      |
| 5     | 1        |
| 10    | 1        |
| 50    | 1        |
| 100   | 1        |

When **C=1**, the accuracy is 0.9 and the plot is as follows:

<div align=left><img src="C:\git\559\HW8\pic\1a1.png" style="zoom:75%;" />

<div align=left><img src="C:\git\559\HW8\pic\1a2.png" style="zoom:75%;" />

When **C=100**, the accuracy is 1 and the plot is as follows:

<div align=left><img src="C:\git\559\HW8\pic\1a3.png" style="zoom:75%;" />

<div align=left><img src="C:\git\559\HW8\pic\1a4.png" style="zoom:75%;" />

**Explanation**: The soft-margin gets larger as the slack parameter **C** decreases and this give rise to more misclassified data points. As we increases **C**, the margin gets smaller and it ends up with less misclassified data points.

### (b):

When **C=100**, the plot looks like the following ad the support vectors are circled with blue.

<div align=left><img src="C:\git\559\HW8\pic\1b1.png" style="zoom:75%;" />

The accuracy, weight vector and decision boundary equation are:

<div align=left><img src="C:\git\559\HW8\pic\1b2.png" style="zoom:75%;" />

### (c):

Plug in the individual support vectors to the decision boundary equation and we have the respective g(x)  values are **[-1, -1, 0.589]**. So we can say that the first two values are on the boundary (correctly classified) and the last one is in the margin (misclassified).

<div align=left><img src="C:\git\559\HW8\pic\1c1.png" style="zoom:75%;" />

### (d):

With RBF kernel and **C=50**, we obtain the **accuracy=0.95**.

<div align=left><img src="C:\git\559\HW8\pic\1d1.png" style="zoom:75%;" />

<div align=left><img src="C:\git\559\HW8\pic\1d2.png" style="zoom:75%;" />

With RBF kernel and **C=5000**, we obtain the **accuracy=1**.

<div align=left><img src="C:\git\559\HW8\pic\1d3.png" style="zoom:75%;" />

<div align=left><img src="C:\git\559\HW8\pic\1d4.png" style="zoom:75%;" />

**Explanation**:  both **C=50** and **C=5000** come up with non-linear boundaries because the feature spaces are initially mapped to higher dimensions and then reduced into 2D. 

As the **C** gets larger, the classification gets more rigorous and thus the boundary gets more fitting. This improvement comes along with the increasing of the model complexity.

### (e):

When **C='default', gamma=10**:

<div align=left><img src="C:\git\559\HW8\pic\1e1.png" style="zoom:75%;" />

<div align=left><img src="C:\git\559\HW8\pic\1e2.png" style="zoom:75%;" />

When **C='default', gamma=100**:

<div align=left><img src="C:\git\559\HW8\pic\1e3.png" style="zoom:75%;" />

<div align=left><img src="C:\git\559\HW8\pic\1e4.png" style="zoom:75%;" />

When **C='default', gamma=1000**:

<div align=left><img src="C:\git\559\HW8\pic\1e5.png" style="zoom:75%;" />

<div align=left><img src="C:\git\559\HW8\pic\1e6.png" style="zoom:75%;" />

**Comment**: For the same **C** and smaller **gamma** value, the classification is more tolerable with the misclassified data and the edges of the margin are more flat and smooth. As the **gamma** increases, the edge of the margin gets more complicated and rigorous with the misclassified data.

When **gamma=50**, the accuracy is not so desirable and not fitting very well. When **gamma=500**, the accuracy is ideal and the generalization capacity is great as well. However, as **gamma** increases to **5000**, there exists overfitting problem and the generalization capacity is reduced.



---

## Problem 2:

### (a):

With RBF kernel, **γ=1, C=1**, the average cross-validation accuracy is **0.8203**:

<div align=left><img src="C:\git\559\HW8\pic\2a1.png" style="zoom:75%;" />

### (b):

(i) 

Visualization of ACC, the X-axis represents c and the Y-axis represents gamma.

<div align=left><img src="C:\git\559\HW8\pic\2b1.png" style="zoom:75%;" />

(ii)

The criterion picking the best value is to choose the pair with the largest accuracy and when there are multiple pairs who share the same largest accuracy, we pick the one with the least deviation.

Here, we pick the best pair is **C=8.286 γ=0.869"** and the **best accuracy is 0.866** and **deviation is 0.08**.

<div align=left><img src="C:\git\559\HW8\pic\2b2.png" style="zoom:75%;" />

### (c):

(i)

The values of the 20 chosen pairs of **C** and **γ** are:

<div align=left><img src="C:\git\559\HW8\pic\2c1.png" style="zoom:55%;" />

The corresponding **accuracy and deviation** are:

<div align=left><img src="C:\git\559\HW8\pic\2c2.png" style="zoom:65%;" />

(ii)

**Comment**: For each run over the 20 iterations, the criterion picking the best value is to choose the pair with the largest accuracy and when there are multiple pairs who share the same largest accuracy, we pick the one with the least deviation. 

The pick of best values runs more **reproducible** because theoretically, the SVM problem is a convex optimization and with strong duality property. The result of the dual optimization should be the global optimal result. So the parameter can converge and the pair of best **C** and **γ** may be more reproducible.

The best pair of **C** and **γ** is **[1.93069773e+01 3.72759372e-01]**

The corresponding **accuracy** and **deviation** is **[0.87777778 0.1237281]**

### (d):

The classification accuracy on the test set is **0.8314**. This value is between **0.8778±0.1237** so it is in the one standard deviation from the cross-validation.

<div align=left><img src="C:\git\559\HW8\pic\2d2.png" style="zoom:65%;" />

<div align=left><img src="C:\git\559\HW8\pic\2d1.png" style="zoom:75%;" />



