---
layout: post
title: "Elevating Credit Card Approvals: A Dual Approach with SVM and KNN Models for Eligibility Assessment"
subtitle: "An exploration of SVM and KNN models to determine the ideal credit card candidacy"
# date: 2022-07-15 23:45:13 -0400
background: '/img/post/post2/header_img.png'
---

## SVM Classifier - Linear 

The main goal of this project is to develop a good SVM classifier using the credit_card_data.txt data set. In order to identify the best performing SVM classifier, the key hyperparameter: $$\lambda$$, has to be identified. 

In the case of this exercise, the value $$\lambda$$, will here after be referred to as `C`. The code presented below will demonstrate the methodology utilized to determine the best value for this hyperparameter.    

```r
# Importing the necessary library for analysis
library(kernlab)

# Setting up the working directory data
getwd()
setwd("../data 2.2")

```

Having imported the necessary libraries and setup the working directory, the data set required for this analysis was called and stored within the `credit_card_dataH` variable. The data set was also converted into a matrix datatype as this is the only datatype accepted by the `ksvm()` function.     

```r
### initializing a variable to store the data
credit_card_dataH <-
  as.matrix(read.delim("credit_card_data-headers.txt"))

```

To identify the best `C` value, a trail and error approach was initially taken. In this approach, the values used for `C` included: 

* C = 1
* C = 10
* C = 100

The accuracy of the model for each of the `C` values was calculated by dividing the **sum** of the total number of correctly classified values by the model over the total number of correctly classified observations.      

The results for these hyperparameters are presented below; using the following code. 

```r
# for C = 1
creditcard_model_1 <-
  ksvm(
    credit_card_dataH[, 1:10],
    credit_card_dataH[, 11],
    type = "C-svc",
    kernel = "vanilladot",
    C = 1,
    scaled = TRUE
  )

model_prediction_1 <-
  predict(creditcard_model_1, credit_card_dataH[, 1:10])

acc_value_1 <-
  sum(model_prediction_1 == credit_card_dataH[, 11]) / nrow(credit_card_dataH)

creditcard_model_2 <-
  ksvm(
    credit_card_dataH[, 1:10],
    credit_card_dataH[, 11],
    type = "C-svc",
    kernel = "vanilladot",
    C = 10,
    scaled = TRUE
  )

model_prediction_2 <-
  predict(creditcard_model_2, credit_card_dataH[, 1:10])

acc_value_2 <-
  sum(model_prediction_2 == credit_card_dataH[, 11]) / nrow(credit_card_dataH)

creditcard_model_3 <-
  ksvm(
    credit_card_dataH[, 1:10],
    credit_card_dataH[, 11],
    type = "C-svc",
    kernel = "vanilladot",
    C = 100,
    scaled = TRUE
  )

model_prediction_3 <-
  predict(creditcard_model_3, credit_card_dataH[, 1:10])

acc_value_3 <-
  sum(model_prediction_3 == credit_card_dataH[, 11]) / nrow(credit_card_dataH)

print(paste0("Accuracy = ", round(acc_value_1, 6)*100, "% when C = 1"))
print(paste0("Accuracy = ", round(acc_value_2, 6)*100, "% when C = 10"))
print(paste0("Accuracy = ", round(acc_value_3, 6)*100, "% when C = 100"))
```
Since the accuracy value is the same for all 3 different `C` values (1, 10, 100), another approach was taken to test if an even higher accuracy could be achieved. Therefore, the `C` values at each order of magnitude in the range from $$1e^{-8}$$ to $$1e^{8}$$ were used to determine `C` value that yielded the best accuracy from the ksvm model. 

A **for** **loop** was devised, that grabbed an element from a list of `C` values from the range specified above. After the `ksvm()` function is executed for every `C` value from the list, the `predict()` function is executed to obtain the values predicted by the model. These predicted values are then plugged into the accuracy formula `acc_value` to determine the performance of the model at each of the `C` values. The `C` value and the associated accuracy value are then subsequently appended to an empty data frame to view the values after. 

The code for the above explanation is presented below: 

```r
# defining a list of C values, to be plugged 
# into the C parameter, within the ksvm() function. 
C_list = as.list(c(-8:8))

# defining an empty data frame to append the 
# final accuracy values for each iteration of C value.  
results_df = data.frame()

# setting up the model within the loop to carry out the calculation process
for (i in seq_along(C_list)) {
  c_val <- C_list[[i]]
  creditcard_model <-
    ksvm(
      credit_card_dataH[, 1:10],
      credit_card_dataH[, 11],
      type = "C-svc",
      kernel = "vanilladot",
      C = 10 ^ c_val,
      scaled = TRUE
    )
  
  model_prediction <-
    predict(creditcard_model, credit_card_dataH[, 1:10])
  
  acc_value <-
    sum(model_prediction == credit_card_dataH[, 11]) / nrow(credit_card_dataH)
  
  # appending the results to the "output" list 
  output = c(paste("1e", c_val, sep = ""), round(acc_value, 10)*100)
  
  # appending the output list to the results_df data frame
  results_df = rbind(results_df, output)
}

# naming the columns of the results data frame, that contains the accuracy
# values obtained from the for loop above.
colnames(results_df) <- c("C_value", "accuracy_value (%)")

# printing the data frame
results_df
```


Based on the results observed in the `results_df` data frame, the same accuracy value is obtained for `C` values in the range from $$1e^{-2}$$ to $$1e^{2}$$. As such, the median value of this range $$1e^{0}$$, or C = 1 is determined to be the best value. 

Therefore, using this value (C = 1), the coefficients $$a_{0}$$ and $$a_{1}...a_{m}$$ will be calculated using the code presented below:

```r
# running back the model with the most accurate C value
# to obtain the coefficients for the SVM model
creditcard_model <-
  ksvm(
    credit_card_dataH[, 1:10],
    credit_card_dataH[, 11],
    type = "C-svc",
    kernel = "vanilladot",
    C = 1,
    scaled = TRUE
  )

# calculating the coefficients a1...a15
a <-
  colSums(creditcard_model@xmatrix[[1]] * creditcard_model@coef[[1]])

# calculating the intercept a0
a0 <- creditcard_model@b

a

a0
```


Based on the coefficients obtained from the calculations above, the best performing SVM classifier's equation is:

$$0 = -0.00110266416005534 \cdot A_{1} -0.000898053885177352 \cdot A_{2} -0.00160745568843068 \cdot$$ 
$$ A_{3} +0.00290416995926498 \cdot A_{8} +1.00473634563239 \cdot A_{9} -0.00298521097400601 \cdot$$ 
$$ A_{10} -0.000203517947504475 \cdot A_{11} -0.000550480305885316 \cdot A_{12} -0.0012519186641109 \cdot$$ 
$$ A_{14} +0.1064404601442\cdot A_{15} -0.0814838195688614 \cdot A_{0}$$

with a prediction accuracy of: **86.3914%**.

## SVM Classifier - Non Linear

Similar to the approach taken in Question 2.2.1, the approach here will also be the same. However, in the previous question, the model obtained provides a linear (or multiple linear) equation. In this case, a non-linear equation will be obtained. 

What differentiates the model in this scenario, is changing the value of a single parameter within the `ksvm()` function. This would be the `kernel` parameter. Where it was previously set to `vanilladot`, here this parameter will be set to `rbfdot`, which as per the documentation provided stands for Radial Basis kernel "Gaussian". Additionally, the `polydot` (Polynomial kernel) will also be applied to compare the performance of the model against the `rbfdot` kernel.

First, the results for the `rbfdot` kernel will be presented. Please note that the code used here is same as the code presented in the previous question

```r

### initializing a variable to store the data
credit_card_dataH <-
  as.matrix(read.delim("credit_card_data-headers.txt"))

# defining a list of C values, to be plugged 
# into the C parameter, within the ksvm() function. 
C_list_rbfdot = as.list(c(-8:8))

# defining an empty data frame to append the 
# final accuracy values for each iteration of C value.  
results_df_rbfdot = data.frame()

# setting up the model within the loop to carry out the calculation process
for (i in seq_along(C_list_rbfdot)) {
  c_val <- C_list_rbfdot[[i]]
  creditcard_model_rbfdot <-
    ksvm(
      credit_card_dataH[, 1:10],
      credit_card_dataH[, 11],
      type = "C-svc",
      kernel = "rbfdot",
      C = 10 ^ c_val,
      scaled = TRUE
    )
  
  model_prediction <-
    predict(creditcard_model_rbfdot, credit_card_dataH[, 1:10])
  
  acc_value <-
    sum(model_prediction == credit_card_dataH[, 11]) / nrow(credit_card_dataH)
  
  # appending the results to the "output" list 
  output = c(paste("1e", c_val, sep = ""), round(acc_value, 10)*100)
  
  # appending the output list to the results_df_rbfdot data frame
  results_df_rbfdot = rbind(results_df_rbfdot, output)
}

# naming the columns of the results data frame, that contains the accuracy
# values obtained from the for loop above.
colnames(results_df_rbfdot) <- c("C_value", "accuracy_value (%)")

# printing the data frame
results_df_rbfdot
```

The results for the `polydot` kernel are now presented below 

```r

### initializing a variable to store the data
credit_card_dataH <-
  as.matrix(read.delim("credit_card_data-headers.txt"))

# defining a list of C values, to be plugged 
# into the C parameter, within the ksvm() function. 
C_list_polydot = as.list(c(-8:8))

# defining an empty data frame to append the 
# final accuracy values for each iteration of C value.  
results_df_polydot = data.frame()

# setting up the model within the loop to carry out the calculation process
for (i in seq_along(C_list_polydot)) {
  c_val <- C_list_polydot[[i]]
  creditcard_model_polydot <-
    ksvm(
      credit_card_dataH[, 1:10],
      credit_card_dataH[, 11],
      type = "C-svc",
      kernel = "rbfdot",
      C = 10 ^ c_val,
      scaled = TRUE
    )
  
  model_prediction <-
    predict(creditcard_model_polydot, credit_card_dataH[, 1:10])
  
  acc_value <-
    sum(model_prediction == credit_card_dataH[, 11]) / nrow(credit_card_dataH)
  
  # appending the results to the "output" list 
  output = c(paste("1e", c_val, sep = ""), round(acc_value, 10)*100)
  
  # appending the output list to the results_df_rbfdot data frame
  results_df_polydot = rbind(results_df_polydot, output)
}

# naming the columns of the results data frame, that contains the accuracy
# values obtained from the for loop above.
colnames(results_df_polydot) <- c("C_value", "accuracy_value (%)")

# printing the data frame
results_df_polydot
```

Based on the results observed for the models above, the accuracy values starting from $$1e^{2}$$ and upwards must be treated with caution since the entire data set was used for training the model, which would very likely result in the model overfitting to all the data points, thus leading to such high accuracy values. 

The best way to further determine the accuracy of the non-linear models here would be to split the data set into a training/validation/test data sets.


## KNN Approach 

For this question, the k-nearest neighbor (knn) algorithm was used to classify the data points in this data set. 

In order to obtain the best k value, a **nested** **for** **loop** was devised. For every value of k (the outer loop), in a defined range, a nested for loop iterated over all the rows (the inner loop), except for one row **`i`**, of the `credit_card_data.txt` data frame. The excluded `ith` row is used as the test data point to determine the accuracy of the knn model for each of the k values in the defined range. 

The accuracy of the knn model is determined using the same accuracy formula used in question 2.2.1, wherein the accuracy of the knn model is determined dividing by using the **sum** of the total number of correctly classified values by the knn model over the total number of correctly classified observations.  

```r
# Importing the necessary libraries for analysis
library(kknn)
library(tidyverse)

# Setting up the working directory data
getwd()
setwd("../data 2.2")
```


```r
# reading in the data set into a new data frame
creditcard_df <-
  read.table(file = "credit_card_data.txt", sep = "\t")
```


Initially, an empty data frame is initialized to store the final results of the analysis. This data frame is setup to store the k-values and the associated accuracy values for each iteration of the k value, defined as `kval` in the code below. 

An empty array, with the same length as the `credit_card_data.txt` data set is initialized to temporarily store the predictions made by the `kknn()` function; to be used in calculating the accuracy of the model for each iteration of the `kval` value.

The output of the `kknn()` function is a continuous value, as opposed to a categorical value. Intrinsically, the output of the model should be binary (categorical), since that is the response variable; a data point either belongs to a class or it does not. With the continuous value the `kknn()` function produces however, a fraction value is yielded, which essentially communicates that a "fraction" of k belongs to a particular class. Since this is not an ideal approach to determine the best k-value; this "fraction" value or the "continuous" value is rounded to the nearest factor. This is represented by the code snippet:

`kknn_predictions[i] <- as.integer(fitted(creditcard_knn) + 0.5)`

where, the continuous value is rounded off to 0 or 1 by a threshold factor of 0.5. The threshold factor **0.5** was determined through a brute force approach, where values in the range from 0.1 to 1.0 (+0.1 incrementals) were manually entered to determine the best value for `k`. Please see Table 1, in the appendix section to view all the k-values obtained for each of the threshold factors. 

Having obtained the best threshold value, the accuracy of the knn model for each k-value was then calculated, and appended to the `knn_results_df` data frame to view the final results. 

```r
# initialzing an empty data frame
knn_results_df = data.frame()

# initializing an empty array to store the knn prediction values 
kknn_predictions <- array(0, nrow(creditcard_df))

for (kval in 1:40) {
  for (i in 1:nrow(creditcard_df)) {
    creditcard_knn <-
      kknn(V11 ~ .,
           creditcard_df[-i,],
           creditcard_df[i,],
           k = kval,
           distance = 2,
           kernel = "optimal",
           scale = TRUE)
    kknn_predictions[i] <- as.integer(fitted(creditcard_knn) + 0.5)
  }
  accuracy = sum(kknn_predictions == creditcard_df[,11]) / nrow(creditcard_df)
  
  # appending the results to the "output" list 
  output = c(kval, round(accuracy, 10)*100)
  
  # appending the output list to the results_df data frame
  knn_results_df = rbind(knn_results_df, output)
}

# naming the columns for the knn_results_df dataframe. 
colnames(knn_results_df) <- c("k_value", "accuracy_value")

# printing the accuracy values for the predictions made by the knn model
knn_results_df
```


The k-value with the highest accuracy is filtered and presented in the table below: 

```r
knn_results_df %>% slice_max(accuracy_value)
```

The best `k_value` based on the analysis performed here is, **k = 12**, and has the best accuracy level of: 85.32%

The rationale behind picking the value 12, and not 15 is because, when identifying the best threshold factor, two factors (0.5 and 0.6) were evaluated further, before **0.5** was picked as the final value. This is because, the `k_value` = 12 was obtained both times; when the threshold value was set to 0.5, and when it set was 0.6. Additionally, picking a larger k-value might incur the possibility of miscategorizing a data point into a wrong class.   



## Appendix

Table 1. The table below shows the threshold_values and the corresponding k_values that yielded the highest accuracy.   

```r
threshold_value <- c(seq(0.1:1.0, by = 0.1))
k_value <-
  c(
    "1",
    "1, 2",
    "24",
    "27",
    "12, 15",
    "8, 12",
    "17, 18, 19, 20, 21, 22",
    "31, 32",
    "1",
    "36, 37, 38, 39, 40"
  )

df <- data.frame(threshold_value, k_value)

print(df)
```