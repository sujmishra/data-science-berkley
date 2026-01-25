# Methodology 
 We will use CRISP-DM or cross Industry Standard Process for Data Mining as the method of data analysis.

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


# Understanding the Business

This research can help address a significant public health problem. If it can be shown that lifestyle factors (which can be modified) have a significant contribution in predicting risk of diabetes, then public health programs can be targeted towards certain empowering people to control diabetes through lifestyle changes.

As per https://www.cdc.gov/diabetes/php/data-research/index.html 40.1 million people(12% of the population) are diabetic. Since a very large population of 115.2 million are pre-diabetic , targeted intervention can help contain this risk. This will could reduce the burden of this disease on Americans.

# Understanding the Data

The dataset contains 253680 rows and  22 columns. The dataset contains 21 feature columns and 1 target column.

## Feature columns 

| Name  	                    | Type                 |    Description                                               |
|----------------------------|----------------------|--------------------------------------------------------------|
| HighBP                     | Categorical(Binary)  | Does the respodent have high blood pressure?                 |
| HighChol                   | Categorical(Binary)  | Does the respodent have high cholestrol?                     |
| CholCheck                  | Categorical(Binary)  | Has the respodent done a cholestrol check in past 5 years?   |
| BMI                        | Numeric (Float)      | Body mass Index of the respodent                             |
| Smoker                     | Categorical(Binary)  | Has the respodent smoked atleast 100 cigs in his entire life?|
| Stroke                     | Categorical(Binary)  | Was the respodent ever told that he had a stroke?            |
| HeartDiseaseorAttack       | Categorical(Binary)  | Does the respodent have coronary heart disease?              |
| PhysActivity               | Categorical(Binary)  | Did the respodent do any physical activity in past 30 days?  |
| Fruits                     | Categorical(Binary)  | Does the respodent consume fruits one or more times daily?   |
| Veggies                    | Categorical(Binary)  | Does the respodent consume veggies one or more times daily?  |
| HvyAlcoholConsump          | Categorical(Binary)  | Is the respodent a heavy drinker(14 drinks/week for men and 7 drinks/week for women)                 |
| AnyHealthcare              | Categorical(Binary)  | Does the respodent have healthcare coverage?                 |
| NoDocbcCost                | Categorical(Binary)  | Did the respodent visit a doctor in the past 12 months       |
| GenHlth                    | Categorical(Ordinal) | Respodent rating on a scale of 1-5 on general health         |
| MentHlth                   | Numeric              | Number of days during past 30 days when respodent encountered stress, depression or any other mental health challenges.              |
| PhysHlth                   | Numeric              | Number of days during past 30 days when respodent encountered injuries or other physical health challenges.              |
| DiffWalk                   | Categorical(Binary)  | Does the respodent have difficulty walking or climbing stairs?               |
| Sex                        | Categorical(Binary)  | Gender of the respodent (Male or Female)                 |
| Age                        | Categorical(Ordinal) | Age level of the respodent based on a 13-level age category(_AGEG5YR)  1 = 18-24 9 = 60-64 13 = 80 or older                 |
| Education                  | Categorical(Ordinal) | Education level (EDUCA) of the respodent           |
| Income                     | Categorical(Ordinal) | Income level (INCOME2 )  of the respodent              |

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
> Since no missing values are found, imputing with median or mean for numeric features is not required. Similarly no handling is required for the categorical features.

### Duplicate Rows
The dataset contains **23899** duplicate rows 

After clean-up of the duplicate rows, we are left with 229781 rows.

### Data Visualization

A pie plot of the target column shows the following distribution

| Type  	                    | Count                |    Percentage       |
|----------------------------|----------------------|---------------------|
| No Diabetes                | 190055               |  82.7%              |
| Pre Diabetes               | 4629                 |  2.0%               |
| Diabetes                   | 35097                |  15.3%              |

<img alt="alt_text" width="512px" src="images/diabetes_distribution.png" />


A box plot of the numeric features was created to show the co-relation between the features and the target.

<img alt="alt_text" width="512px" src="images/numeric_boxplots.png" />

The plot shows the following 

1. Corelation of Body Mass Index(BMI) to diabetes 
   - Respodents with "Pre Diabetes" or "Diabetes" have a high BMI
   - As median BMI increases, the risk of falling into "Pre Diabetes" or "Diabetes" increases
> [!NOTE]
> High BMI has a strong corelation with the risk of diabetes.

2. Corelation of mental health to diabetes 
   - Respodents with median days with mental health conditions is close to negligible(almost 0) for all categories
   - Some outliers exist with many people reporting a lot of days when mental health conditions persisted.
> [!NOTE]
> Mental health does not seem to impact different diabetic categories differently.

2. Corelation of physical health to diabetes 
   - Respodents with median days of physical health conditions increases in the "Diabetes" category
   - Some outliers exist.
> [!NOTE]
> Poor physical health shows a weak corelation with the risk of diabetes.

A Violin plot of the numeric features was created to show corelation between the features and the target.

<img alt="alt_text" width="512px" src="images/numeric_violinplots.png" />

The violin plot  shows that BMI is more differentiated across different diabetic categories. Mental health and Physical health 
are skewed towards 0 and the differences are hard to visualize(proving the weak or no corelation).
 
# Modeling 

## Baseline Model

