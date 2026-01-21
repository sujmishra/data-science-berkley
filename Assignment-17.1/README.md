# Problem
We have [data](https://archive.ics.uci.edu/dataset/222/bank+marketing) from direct marketing campaigns(phone calls) from a Portugese banking institution. 

The goal of this project is to perform a comparative evaluation of the following classifiers 

* K Nearest Neighbors 
* Logistic Regression 
* Decision Trees
* Support Vector Machines

During this evaluation the models will be compared on the following criteria 

* Training Time
* Training Accuracy 
* Test Accuracy 

Aditionally hyperparameter tuning of the models will be performed in order to adjust and gain the best performance metrics.

# Business Objective 
The dataset contains data collected using 17 directed marketing campaigns using a self owned contact center between May 2008 and November 2010. The dominant marketing channel was telephone , however some of the data was captured using internet banking channel.
A total of 79354 contacts were made and an attractive long-term deposit application, with good interest rates, was offered. For each contact, 20 input attributes and 1 output attribute were stored. For the whole database considered, there were 6499
successes (8% success rate).

The business goal of this problem is to predict if the customer will subscribe to a term deposit based on the marketing campaign and what factors contribute towards the acceptance of a subscription. The marketing campaigns could then be targeted towards individuals with factors that are more likely to subscribe to the term deposit.

We take several models and train them with a part of the dataset. We try to find out which model performs the best in terms of training and testing accuracy. Aditionally we also tune the hyperparameters of the  models to see if the accuracy could be improved.

## Data Preparation

1. Verifying the data revealed that although many columns didnt have NA, they had unknowns
2. The ouput/target column was renamed from y to subscribed for convenience 
3. A feature set was created from banking information features('age','job','marital','education','default','housing','loan')
4. A LabelEncoder was applied on the target column to encode the values. 
5. A pre-processor was created with the following steps 
   - Standard Scaler for numeric features 
   - OneHotEncoder for the categorical features 
   - OrdinalEncoder for the binary features
6. Using train_test_split , the data was split into 80% training data and 20% test data.


# Findings 

## Baseline Model

Using the training and test data a baseline was created using a DummyClassifier with a Most Frequent Strategy 

The Baseline model had the following characteristics 

| Property  	             | Value  |
|----------------------------|--------|
|Training Time (Seconds)     | 0.0037 |
|Baseline training accuracy  | 0.8876 |
|Baseline testing accuracy   | 0.8865 |

## Logistic Regression Model 

| Property  	             | Value  |
|----------------------------|--------|
|Training Time (Seconds)     | 0.0875 |
|Model training accuracy     | 0.5920 |
|Model testing accuracy      | 0.5961 |
|Model Recall Score          | 0.6235 |
|Model F1 Score              | 0.2595 |


> [!NOTE]
> Running the LogisticRegression with default parameters showed the same accuracy as the baseline model. This shows a high degree of imbalance. The above numbers were obtained using a balanced class weight.

## Model Comparision
| Model  	                 | Training Time (Seconds) | Training Accuracy | Testing Accuracy|
|----------------------------|-------------------------|-------------------|-----------------|
|Logistic Regression         | 0.065831	               |   0.887557	       | 0.886502        |
|KNN                         | 0.018772                |   0.890258	       | 0.872785        |
|Decision Tree               | 0.132741	               |   0.916601	       | 0.862952        |
|SVM                         | 19.391068               |   0.888285        | 0.886623        |


Based on these observations 

* KNearestNeighbors had the best training time.
* Decison Tree had the highest training accuracy.
* SVM had the highest testing accuracy followed closely by  Logistic Regression.

## Model Tuning 

Using a GridSearchCV the hyperparameters were tuned for the following models 

| Model  	                 | Training Accuracy       | Testing Accuracy  | Hyperparameters |
|----------------------------|-------------------------|-------------------|-----------------|
|Logistic Regression         | 0.887557		           |   0.886502	       | {'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs'}        |
|KNN                         | 0.889772                |   0.883103	       | {'n_neighbors': 9, 'p': 1, 'weights': 'uniform'}        |
|Decision Tree               | 0.887557		           |   0.886502	       | {'max_depth': 3, 'min_samples_leaf': 1,'min_samples_split': 2}      |
|SVM                         | 0.892868		           |   0.88956	       | {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}      |

> [!NOTE]
> Running GridSearchCV on the SVM made the execution run for a very long time. So the shorter form of the dataset was used 
to train the SVM model.

> [!IMPORTANT]
> Tuning hyperparameters showed improvement in training and testing accuracy for all models.


# Reccomendations 
