# Methodology 
 We will use CRISP-DM or Cross Industry Standard Process For Data Mining as the method of data analysis.

 <img alt="alt_text" width="512px" src="images/CRISP-DM_Process_Diagram.png" />

 CRISP DM contains the following steps 

 1. Understanding the Business
 2. Understanding the Data 
 3. Data Preparation
 4. Modeling 
 5. Evaluation 
 6. Deployment 


## Research Question
**Which lifestyle factors such as diet exercise etc can be used to predict the 
risk of diabetes?**

## Notebook
1. The jupyter notebook used for this exploratory data analysis is present [here](capstone-eda.ipynb) 
2. The jupyter notebook used for model training and tuning is [here](diabetes_models.ipynb) 

# Understanding the Business

This research can help address a significant public health problem. If it can be shown that lifestyle factors (which can be modified) have a significant contribution in predicting risk of diabetes, then public health programs can be targeted towards certain empowering people to control diabetes through lifestyle changes.

As per https://www.cdc.gov/diabetes/php/data-research/index.html 40.1 million people(12% of the population) are diabetic. Since a very large population of 115.2 million are pre-diabetic , targeted intervention can help contain this risk. This will could reduce the burden of this disease on Americans.

**Public health officials, Researchers and general population could be the target audience of this analysis.**

# Understanding the Data

## Dataset

The dataset used here is the [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) dataset from UC Irvine Machine Learning Repository. The UC Irvine Machine Learning Repository points to the [Dataset Home Page](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) from kaggle. The raw data comes from the Behavioral Risk Factor Surveilance System from CDC. We will use the CSV from kaggle to perform the exploratory data analysis

The dataset contains **253680** rows and  **22** columns. The dataset contains 21 feature columns and 1 target column.

## Feature columns 

| Name  	                    | Data Type            | Indicator Type| Description                                  |
|----------------------------|----------------------|---------------|-----------------------------------------------|
| HighBP                     | Categorical(Binary)  | Clinical |Does the respondent have high blood pressure?                 |
| HighChol                   | Categorical(Binary)  | Clinical |Does the respondent have high cholestrol?                     |
| CholCheck                  | Categorical(Binary)  | Clinical |Has the respondent done a cholestrol check in past 5 years?   |
| BMI                        | Numeric (Float)      | Lifestyle| Body mass Index of the respodent                             |
| Smoker                     | Categorical(Binary)  | Lifestyle|Has the respondent smoked atleast 100 cigs in his entire life?|
| Stroke                     | Categorical(Binary)  | Clinical |Was the respondent ever told that he had a stroke?            |
| HeartDiseaseorAttack       | Categorical(Binary)  | Clinical |Does the respondent have coronary heart disease?              |
| PhysActivity               | Categorical(Binary)  | Lifestyle| Did the respondent do any physical activity in past 30 days?  |
| Fruits                     | Categorical(Binary)  | Lifestyle|Does the respondent consume fruits one or more times daily?   |
| Veggies                    | Categorical(Binary)  | Lifestyle|Does the respondent consume veggies one or more times daily?  |
| HvyAlcoholConsump          | Categorical(Binary)  | Lifestyle|Is the respondent a heavy drinker(14 drinks/week for men and 7 drinks/week for women 0 is heavy and 1 is not heavy)                 |
| AnyHealthcare              | Categorical(Binary)  | Lifestyle|Does the respondent have healthcare coverage?                 |
| NoDocbcCost                | Categorical(Binary)  | Lifestyle|Did the respondent not visit a doctor in the past 12 months due to cost?      |
| GenHlth                    | Categorical(Ordinal) | Lifestyle|respondent rating on a scale of 1-5 on general health         |
| MentHlth                   | Numeric              | Lifestyle|Count of number of days during past 30 days when respondent encountered stress, depression or any other mental health challenges.              |
| PhysHlth                   | Numeric              | Lifestyle|Count of number of days during past 30 days when respondent encountered injuries or other physical health challenges.              |
| DiffWalk                   | Categorical(Binary)  | Clinical|Does the respondent have difficulty walking or climbing stairs?               |
| Sex                        | Categorical(Binary)  | Clinical|Gender of the respondent (Male or Female)                 |
| Age                        | Categorical(Ordinal) | Clinical|Age level of the respondent based on a 13-level age category(_AGEG5YR)  1 = 18-24 9 = 60-64 13 = 80 or older                 |
| Education                  | Categorical(Ordinal) | Lifestyle|Education level (EDUCA) of the respondent           |
| Income                     | Categorical(Ordinal) | Lifestyle|Income level (INCOME2 )  of the respondent              |



## Target column

The target column is 'Diabetes_012' which contains values representing the following 
1. 0 means no diabetes. 
2. 1 means pre-diabetes.
3. 2 means diabetes. 

# Data Preparation

## Data Quality 

### Missing values 
The dataset contains no missing or null values (NAN, None or NAT) values

> [!NOTE]
> **Since no missing values are found, imputing with median or mean for numeric features is not required. Similarly no handling is required for the categorical features.**

### Duplicate Rows
The dataset contains **23899** duplicate rows 

After clean-up of the duplicate rows, we are left with **229781** rows.

### Categorical data validation
All categorical columns contain data with right cardinality as defined.


## Data Visualization

A pie plot of the target column shows the following distribution

| Type  	                    | Count                |    Percentage       |
|----------------------------|----------------------|---------------------|
| No Diabetes                | 190055               |  82.7%              |
| Pre Diabetes               | 4629                 |  2.0%               |
| Diabetes                   | 35097                |  15.3%              |

<img alt="alt_text" width="512px" src="images/diabetes_distribution.png" />

Histograms of all features can be shown in the following image 

<img alt="alt_text" width="512px" src="images/histograms.png" />

### Numeric Features
A box plot of the numeric features was created to show the co-relation between the features and the target.

<img alt="alt_text" width="512px" src="images/numeric_boxplots.png" />

The plot shows the following 

1. Correlation of Body Mass Index(BMI) to diabetes 
   - Respodents with "Pre Diabetes" or "Diabetes" have a high BMI
   - As median BMI increases, the risk of falling into "Pre Diabetes" or "Diabetes" increases
> [!NOTE]
> **High BMI has a strong correlation with the risk of diabetes.**

2. Correlation of mental health to diabetes 
   - Respodents with median days with mental health conditions is close to negligible(almost 0) for all categories
   - Some outliers exist with many people reporting a lot of days when mental health conditions persisted.
> [!NOTE]
> **Mental health does not seem to impact different diabetic categories differently.**

2. Correlation of physical health to diabetes 
   - Respodents with median days of physical health conditions increases in the "Diabetes" category
   - Some outliers exist.
> [!NOTE]
> **Poor physical health shows a weak correlation with the risk of diabetes.**

A Violin plot of the numeric features was created to show correlation between the features and the target.

<img alt="alt_text" width="512px" src="images/numeric_violinplots.png" />

The violin plot  shows that BMI is more differentiated across different diabetic categories. Mental health and Physical health are skewed towards 0 and the differences are hard to visualize(proving the weak or no correlation).

A pair wise grid plot of the numeric features is shown here 

<img alt="alt_text" width="512px" src="images/numeric_hexbin.png" />

1. **BMI vs Mental Health**: Most data concentrated at MentHlth =0 across all BMIs. Higher BMI seems slighlty more associated with mental health issues.
2. **BMI vs Physical Health**: Positive correlation is visible. Higher BMI seems to associate with greater physical health issues.
3. **Mental Health vs Physical Health**: A stronger grid pattern is observed here. Positive correlation is observed signalling this respodents with poor physical health also have poor mental health. 

#### IQR Analysis
| Feature  	                    | IQR                |    Outlier Percentage|
|----------------------------|-----------------------|---------------------|
| BMI                        | 6.0                   | 15.3%               |
| Mental Health              | 0.0                   | 17.0%               |
| Physical Health            | 0.0                   | 17.7%               |

> [!NOTE]
> **Outliers in BMI, MentHlth and PhysHlth are retained as they represent valid extreme values in health data. These outliers are clinically meaningful (e.g., very high BMI, frequent poor health days) and should not be removed.**

### Categorical Features
A count plot of the categorical features was created to show the co-relation between the features and the target.

<img alt="alt_text" width="512px" src="images/categorical_countplot.png" />

The plot shows the following

1. **High Blood Pressure (HighBP)** - People with high blood pressure have a significantly higher proportion of diabetes and pre-diabetes.

> [!NOTE]
> **High blood pressure shows a strong positive correlation with diabetes risk.**


2. **High Cholesterol (HighChol)** - Similar to HighBP, people with high cholesterol have higher diabetes prevalence.

> [!NOTE]
> **High cholesterol shows a strong positive correlation with diabetes risk.**


3. **Cholesterol Check (CholCheck)** - People who have had cholesterol checks show higher diabetes rates. This might indicate health-conscious behavior or existing health concerns.

4. **Smoker** - Smoking status shows moderate correlation with diabetes. Smokers have slightly higher diabetes rates.

5. **Stroke** - People who have had strokes show significantly higher diabetes prevalence. Strong association between cardiovascular events and diabetes.

6. **Heart Disease or Attack** - Similar to stroke, strong correlation with diabetes.

7. **Physical Activity (PhysActivity)** - People who do NOT engage in physical activity have higher diabetes rates.

> [!NOTE]
> **Physical inactivity shows strong correlation with increased diabetes risk.**


8. **Fruits** - People who consume fruits daily have lower diabetes rates.

9. **Veggies** - Similar to fruits, vegetable consumption is associated with lower diabetes risk.

10. **Heavy Alcohol Consumption (HvyAlcoholConsump)** - Heavy drinkers show different diabetes patterns. Relationship is complex and requires further investigation.

11. **Healthcare Coverage (AnyHealthcare)** - People with healthcare coverage show higher diabetes diagnosis rates. This could be due to better detection rather than higher actual prevalence.

12. **No Doctor Due to Cost (NoDocbcCost)** - People who skip doctor visits due to cost show different diabetes patterns.

13. **General Health (GenHlth)** - Strong correlation: As self-reported health worsens (1=excellent → 5=poor), diabetes prevalence increases dramatically.

> [!IMPORTANT]
> **Self-reported general health is one of the strongest predictors of diabetes status.**


14. **Difficulty Walking (DiffWalk)** - People with walking difficulties show much higher diabetes rates.

15. **Sex** - Males show slightly higher diabetes prevalence than females.

16. **Age** - Clear trend: Diabetes prevalence increases with age. Strongest increase observed after age category 7 (approximately 45-50 years).

> [!IMPORTANT]
> **Age is a critical factor - diabetes risk increases significantly with age.**


17. **Education** - Lower education levels correlate with higher diabetes rates. Suggests socioeconomic factors play a role.

18. **Income** - Lower income levels show higher diabetes prevalence. Another indicator of socioeconomic influence.

> [!NOTE]
> **Socioeconomic factors (education and income) show inverse correlation with diabetes risk.**


## Correlation Analysis

### Correlation Heatmap
A correlation heatmap was created to identify relationships between features.

<img alt="alt_text" width="512px" src="images/correlation_heatmap.png" />

**Key Findings:**

**Strong Positive Correlations:**
1. **HighBP ↔ HighChol (0.43)**: People with high blood pressure often have high cholesterol.
2. **PhysHlth ↔ DiffWalk (0.50)**: Physical health issues correlate with walking difficulties.
3. **Age ↔ Multiple features**: Age shows moderate positive correlation with many health issues.

**Strong Negative Correlations:**
1. **GenHlth ↔ PhysActivity (-0.28)**: People with poor general health tend to be less physically active.
2. **Income ↔ NoDocbcCost (-0.31)**: Higher income correlates with fewer skipped doctor visits.

**Diabetes Correlations:**
1. **Strongest positive**: HighBP (0.33), HighChol (0.28), BMI (0.28), GenHlth (0.28)
2. **Moderate positive**: Age (0.25), DiffWalk (0.23)
3. **Negative**: PhysActivity (-0.18), Income (-0.12)

> [!IMPORTANT]
> **Clinical markers (HighBP, HighChol, BMI) and self-reported health (GenHlth) are the strongest correlates of diabetes.**


## Feature Engineering
Based on the correlation analysis and domain knowledge, several interaction features were created to capture complex relationships.

### Engineered Features

| Feature Name | Formula | Rationale |
|--------------|---------|-----------|
| **ClinicalMarkerRisk** | HighBP + HighChol + Stroke + HeartDiseaseorAttack + DiffWalk | Cumulative clinical risk score capturing multiple comorbidities |
| **AgeBMIMarkerRisk** | Age × BMI | Interaction between age-related metabolic decline and obesity |
| **BMISedentaryMarkerRisk** | BMI × (1 - PhysActivity) | Combined effect of high BMI and physical inactivity |
| **LifestyleMarkerRisk** | PhysActivity + Fruits + Veggies + (1-HvyAlcoholConsump) + (1-Smoker) | Composite healthy lifestyle score |

> [!NOTE]
> **These engineered features capture important interaction effects that individual features alone cannot represent.**


### Feature Engineering Rationale

1. **ClinicalMarkerRisk**: Medical research shows that patients with multiple comorbidities have exponentially higher diabetes risk than those with single conditions.

2. **AgeBMIMarkerRisk**: The combination of aging (metabolic slowdown) and high BMI creates a multiplicative effect on diabetes risk.

3. **BMISedentaryMarkerRisk**: Sedentary behavior combined with high BMI is particularly dangerous for metabolic health.

4. **LifestyleMarkerRisk**: Captures the cumulative protective effect of multiple positive lifestyle factors.

## Target Variable Transformation

The original target variable `Diabetes_012` has three classes (0, 1, 2). For binary classification, the target was transformed as follows:

**Transformation Applied:**
- **0 (No Diabetes)** → 0
- **1 (Pre-Diabetes)** → 1
- **2 (Diabetes)** → 1

**Rationale:**
- Pre-diabetes and diabetes are both conditions requiring medical intervention
- Combining them creates a clearer screening task: identify anyone at risk
- Matches real-world healthcare screening objectives

**Final Class Distribution:**
- No Diabetes: 190,055 (82.7%)
- Diabetes/Pre-Diabetes: 39,726 (17.3%)

> [!IMPORTANT]
> **Class imbalance (82.7% vs 17.3%) will require special handling during modeling to ensure the model doesn't simply predict "no diabetes" for everyone.**


# Modeling

## Baseline Model Evaluation

Eight models were evaluated as baselines to identify the most promising algorithms for diabetes prediction.

### Models Tested

1. Decision Tree
2. Logistic Regression
3. Support Vector Machine (SVM)
4. Naive Bayes
5. Gradient Boosting
6. Random Forest
7. K-Nearest Neighbors (KNN)
8. AdaBoost

### Baseline Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time (s) |
|-------|----------|-----------|--------|----------|---------|-------------------|
| Decision Tree | 0.708 | 0.309 | 0.761 | 0.440 | 0.797 | 0.38 |
| Logistic Regression | 0.726 | 0.324 | 0.752 | 0.453 | 0.810 | 12.27 |
| SVM (5% sample) | 0.646 | 0.263 | 0.745 | 0.388 | 0.756 | 23.19 |
| Naive Bayes | 0.755 | 0.331 | 0.612 | 0.430 | 0.779 | 0.06 |
| Gradient Boosting | 0.854 | 0.562 | 0.158 | 0.246 | 0.815 | 11.29 |
| Random Forest | 0.843 | 0.441 | 0.152 | 0.226 | 0.772 | 55.45 |
| KNN | 0.784 | 0.358 | 0.456 | 0.401 | 0.717 | 97.23 |
| AdaBoost | 0.852 | 0.546 | 0.193 | 0.286 | 0.800 | 11.94 |

### Key Observations

> [!IMPORTANT]
> **Recall is the critical metric for healthcare screening** - we want to identify as many diabetic patients as possible to minimize false negatives.

**Best Performers by Recall:**
1. **Decision Tree**: 76.1% recall
2. **Logistic Regression**: 75.2% recall
3. **SVM**: 74.5% recall

**Poor Performers:**
- **Gradient Boosting**: 15.8% recall (optimized for accuracy, missed 84% of diabetics)
- **Random Forest**: 15.2% recall (same issue)
- **AdaBoost**: 19.3% recall

> [!WARNING]
> **Ensemble models (Random Forest, Gradient Boosting) failed dramatically on this imbalanced dataset**, achieving high accuracy (85%) but catastrophically low recall (15%). They optimized for the majority class.

## Hyperparameter Tuning

Based on baseline results, Decision Tree and Logistic Regression were selected for hyperparameter tuning due to their superior recall performance.

### Decision Tree Grid Search

| Parameter | Values Tested |
|-----------|---------------|
| max_depth | [8, 10, 12, 15, None] |
| min_samples_split | [10, 20, 30, 50] |
| min_samples_leaf | [5, 10, 15, 20] |
| class_weight | ['balanced', {0:1, 1:8}, {0:1, 1:10}] |
| criterion | ['gini', 'entropy'] |

- **Total Combinations**: 480
- **Cross-Validation**: 5-fold
- **Optimization Metric**: Recall
- **Training Time**: 147 seconds (~2.5 minutes)

### Logistic Regression Grid Search

| Parameter | Values Tested |
|-----------|---------------|
| C | [0.1, 1, 10, 100] |
| penalty | ['l2'] |
| solver | ['liblinear'] |
| class_weight | ['balanced', {0:1, 1:8}, {0:1, 1:10}] |
| max_iter | [5000] |

- **Total Combinations**: 12
- **Cross-Validation**: 5-fold
- **Optimization Metric**: Recall
- **Training Time**: 1442 seconds (~24 minutes)

> [!NOTE]
> **Both models converged on class_weight {0: 1, 1: 10}**, which heavily penalizes false negatives - critical for healthcare screening where missing a diabetic patient has serious consequences.

## Tuned Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time (s) |
|-------|----------|-----------|--------|----------|---------|-------------------|
| Tuned Decision Tree        | 0.592    | 0.256     | **0.890**| 0.397    | 0.805    | 147.02            |
| Tuned Logistic Regression  | 0.620    | 0.268     | **0.875**| 0.410    | 0.810    | 1442.01           |

## Best Hyperparameters

### Tuned Decision Tree
| Parameter              | Best Value      |
|------------------------|-----------------|
| class_weight           | {0: 1, 1: 10}   |
| criterion              | entropy         |
| max_depth              | 8               |
| min_samples_leaf       | 20              |
| min_samples_split      | 50              |

**Training Time**: 2.45 minutes

### Tuned Logistic Regression
| Parameter              | Best Value      |
|------------------------|-----------------|
| C                      | 1               |
| class_weight           | {0: 1, 1: 10}   |
| max_iter               | 5000            |
| penalty                | l2              |
| solver                 | liblinear       |

**Training Time**: 24.03 minutes

## Key Findings

> [!NOTE]
> **Both tuned models achieved excellent recall (87-89%)** - a significant improvement over baseline models (76% recall). This means they now identify nearly 9 out of 10 diabetic patients.

> [!IMPORTANT]
> **Trade-offs:**
> - **Recall improved**: 76% → 89% (Decision Tree baseline to tuned)
> - **Precision decreased**: 31% → 26% (more false positives)

## Feature Importance - Tuned Decision Tree


| Rank | Feature                  | Importance | Category            | Interpretation |
|------|--------------------------|------------|---------------------|----------------|
| 1    | ClinicalMarkerRisk       | 0.4067     | Engineered Feature  | Cumulative clinical risk score (comorbidities) |
| 2    | GenHlth                  | 0.2382     | Lifestyle (Ordinal) | Self-reported general health (1=excellent, 5=poor) |
| 3    | AgeBMIMarkerRisk         | 0.2353     | Engineered Feature  | Age × BMI interaction |
| 4    | BMI                      | 0.0622     | Numeric             | Body Mass Index |
| 5    | CholCheck                | 0.0102     | Clinical (Binary)   | Cholesterol check in past 5 years |
| 6    | HvyAlcoholConsump        | 0.0081     | Lifestyle (Binary)  | Heavy alcohol consumption |
| 7    | Sex                      | 0.0054     | Clinical (Binary)   | Gender |
| 8    | MentHlth                 | 0.0054     | Lifestyle (Numeric) | Days of poor mental health (past 30 days) |
| 9    | Income                   | 0.0048     | Lifestyle (Ordinal) | Income level |
| 10   | HighBP                   | 0.0044     | Clinical (Binary)   | High blood pressure |
| 11   | DiffWalk                 | 0.0040     | Clinical (Binary)   | Difficulty walking/climbing stairs |
| 12   | PhysHlth                 | 0.0033     | Lifestyle (Numeric) | Days of poor physical health (past 30 days) |
| 13   | Education                | 0.0031     | Lifestyle (Ordinal) | Education level |
| 14   | HighChol                 | 0.0019     | Clinical (Binary)   | High cholesterol |
| 15   | Age                      | 0.0016     | Clinical (Ordinal)  | Age category |


The following plot shows the feature importance as determined by the DecisionTreeClassifier. 
<img alt="alt_text" width="512px" src="images/feature_importance_dt.png" />


## Training Neural Networks 


### Simple Architecture 

| Layer | Type | Units/Shape | Activation | Parameters | Purpose |
|-------|------|-------------|------------|------------|---------|
| Input | Input | (21,) | - | 0 | Accepts 21 features (original features, no engineered) |
| Hidden 1 | Dense | 64 | ReLU | 1,408 | First hidden layer for feature learning |
| Hidden 2 | Dense | 32 | ReLU | 2,080 | Second hidden layer for pattern extraction |
| Output | Dense | 1 | Sigmoid | 33 | Binary classification output (diabetes probability) |





### Neural Network Performance Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Accuracy | 0.739 | 73.9% of all predictions were correct |
| Precision | 0.333 | When model predicts diabetes, it's correct 33.3% of the time |
| Recall | **0.709** | Model identifies **70.9%** of diabetic patients |
| F1-Score | 0.453 | Harmonic mean of precision and recall |
| ROC-AUC | 0.802 | Model's ability to distinguish between classes |
| Training Time | 0.91 min | Model trained in under 1 minute |


**Recall Over Training** 

1. Training recall starts at 80% and increases to 82%, showing that the model is learning.
2. Validation recall fluctuates (0.71–0.80).


**Loss Over Training**
1. Training loss decreases from 93% to 86% showing optimization.


### Advanced Architecture


| Layer | Type | Units/Shape | Activation | Regularization | Parameters | Purpose |
|-------|------|-------------|------------|----------------|------------|---------|
| Input | Input | (21,) | - | - | 0 | Accepts 21 original features |
| Hidden 1 | Dense | 256 | ReLU | L2 (0.01) | 5,632 | Wide entry layer for initial feature learning |
| Hidden 2 | Dense | 128 | ReLU | L2 (0.01) | 32,896 | Intermediate pattern extraction |
| Hidden 3 | Dense | 64 | ReLU | L2 (0.01) | 8,256 | Higher-level feature combinations |
| Hidden 4 | Dense | 32 | ReLU | None | 2,080 | Deep pattern recognition |
| Hidden 5 | Dense | 16 | ReLU | None | 528 | Bottleneck layer before output |
| Output | Dense | 1 | Sigmoid | None | 17 | Binary diabetes probability [0,1] |

### Neural Network Performance Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Accuracy | 0.687 | 68.7% of all predictions were correct |
| Precision | 0.305 | When model predicts diabetes, it's correct 30.5% of the time |
| **Recall** | **0.819** | Model identifies **81.9%** of diabetic patients |
| F1-Score | 0.444 | Harmonic mean of precision and recall |
| ROC-AUC | 0.819 | Model's ability to distinguish between classes |
| Training Time | 1.55 min | Model trained in ~90 seconds |


**Recall Over Training**

1. Training recall completely flat at ~81% for all 100 epochs.
2. Validation recall fluctuates between 75-86% (±6%).
3. Zero Learning After Initialization


**Loss Over Training**
1. Training loss decreases flat at 93% showing no optimization.
2. L2 penalty didn't help convergence

The image below shows how the neural network performs
<img alt="alt_text" width="512px" src="images/diabetes_advanced_nn_training.png" />



> [!IMPORTANT]
>  After multiple attempts with 
>  1. 2 layer and 5 layer architecture
>  2. L2 and No regularization
>  3. Class weights
>  4. Fixed and random validation weights
> **We see inconsistent 71-81% recall with Neural Networks vs consistent recall of 89% with DecisionTree and 87.5% with LogisticRegression.**
