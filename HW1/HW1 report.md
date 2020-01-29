# EE559 Homework 1 (week 2)

**Jingquan Yan**

**USC ID: 1071912676**

**Email: jingquan@usc.edu**

---

**(a)** :

> For each of the two synthetic datasets, there are in total C=2 classes and D=2 features.
> For each synthetic dataset: (i) train the classifier, plot the (training-set) data points, the
> resulting class means, decision boundaries, and decision regions (using
> PlotDecBoundaries()); also run the trained classifier to classify the data points from
> their inputs; give the classification error rate on the training set, and separately give the
> classification error rate on the test set. The test-set data points should never be used for
> training. Turn in the plots and error rates.

**Solution:**

1. Classifier plot and error rates for **training-set** 1:

   <div align=left><img src="C:\git\559\HW1\pic\1.png" style="zoom:50%;" />

   <div align=left><img src="C:\git\559\HW1\pic\1-2.png" style="zoom:90%;" />

2. Classifier plot and error rates for **test-set** 1:

   <div align=left><img src="C:\git\559\HW1\pic\2.png" style="zoom:50%;" />
   
   <div align=left><img src="C:\git\559\HW1\pic\2-2.png" style="zoom:90%;" />

3. Classifier plot and error rates for **training-set** 2:

   <div align=left><img src="C:\git\559\HW1\pic\3.png" style="zoom:50%;" />

   <div align=left><img src="C:\git\559\HW1\pic\3-2.png" style="zoom:90%;" />

4. Classifier plot and error rates for **test-set** 2:

   <div align=left><img src="C:\git\559\HW1\pic\4.png" style="zoom:50%;" />

   <div align=left><img src="C:\git\559\HW1\pic\4-2.png" style="zoom:90%;" />

---

**(b)**

> Is there much difference in error rate between the two synthetic datasets? Why or why
> not?

**Answer:**

There is an obvious difference in error rate between the two synthetic datasets. 

The reason is that the decision criteria only brings the average coordinate into consideration and ignores the data distribution (overall shape). Hence the error rate may vary when we have different distribution of input.

---

**(c)**

> For the wine dataset, there are in total C=3 classes (grape cultivars) and D=13 features
> (measured attributes of the wine). In this problem you are to use only 2 features for
> classification.
>
> Pick the first two features (alcohol content and malic acid content), and
> repeat the procedure of part (a) for this dataset.

**Solution:**

1. Classifier plot and error rates for wine **training-set** with column 1 and 2:

   <div align=left><img src="C:\git\559\HW1\pic\5.png" style="zoom:50%;" />

   <div align=left><img src="C:\git\559\HW1\pic\5-2.png" style="zoom:90%;" />

2. Classifier plot and error rates for wine **test-set** with column 1 and 2:

   <div align=left><img src="C:\git\559\HW1\pic\6.png" style="zoom:50%;" />
   
   <div align=left><img src="C:\git\559\HW1\pic\6-2.png" style="zoom:90%;" />

---

**(d)**

> Again for the “wine” dataset, find the 2 features among the 13 that achieve the
> minimum classification error on the training set. (We haven’t yet covered how to do
> feature selection in class, but will later in the semester. For this problem, try coming up
> with your own method - one that you think will give good results - and see how well it
> works.

**Solution:**

1. We use the error rate as the criteria to choose the 2 columns that could make the best classification (least error rate) of the data. After traversal all 78 combinations ($$\frac{{{13}^{2}}-13}{2}$$), the 2 features in **training data** that reaches the minimum error rate are columns 1 and 12 with error rate 0.07866:

   <div align=left><img src="C:\git\559\HW1\pic\7-2.png" style="zoom:90%;" />

   And the corresponding plot is:

   <div align=left><img src="C:\git\559\HW1\pic\7.png" style="zoom:50%;" />

2. To examine the generalization ability of the two columns selected above, we input the corresponding columns in the **test data**, the result is as follows:

   <div align=left><img src="C:\git\559\HW1\pic\8.png" style="zoom:50%;" />
   
   <div align=left><img src="C:\git\559\HW1\pic\8-2.png" style="zoom:90%;" />

---

**(e)**

For **training data**, there are differences for different pairs of features but not such large. The error rates of different pairs are around 0.1-0.5 and the mean and standard deviation are as follow:

<div align=left><img src="C:\git\559\HW1\pic\9-3.png" style="zoom:90%;" />

For **test data**, there are differences as well and the standard deviation of error rate is lower than the training dataset:

<div align=left><img src="C:\git\559\HW1\pic\9-4.png" style="zoom:90%;" />

