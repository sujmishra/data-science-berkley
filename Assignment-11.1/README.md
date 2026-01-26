# Methodology 
 We will use CRISP-DM or Cross Industry Standard Process For Data Mining as the method of data analysis.

 <img alt="alt_text" width="512px" src="images/crisp.png" />

 CRISP DM contains the following steps 

 1. Understanding the Business
 2. Understanding the Data 
 3. Data Preparation
 4. Modeling 
 5. Evaluation 
 6. Deployment 

# Understanding the Business

As per this [forecast](https://www.mordorintelligence.com/industry-reports/united-states-used-car-market), the used car industry in the US is at a staggering 871 billion USD in 2026 and poised to grow to 980.47 billion USD by 2031. 

The key business objective is to understand and identify the important features in a used car that can help sales teams in car dealerships determine the types of used cars that should be kept in inventory to maximize sales. A used car dealership works on the principle that the vehicle they acquire should sell for a profit. Therefore, it's important to be able to predict the final sale price of a car based on its characteristics. 

This can help dealers make informed buying decisions and also determine the price of the used car they purchase for resale. While maximizing profit is one of the important goals, another factor is to identify market demand and purchase the right cars that can be easily resold.

**Target audience: Sales and acquisition teams in used car dealerships**

# Understanding the Data

## Dataset
The provided dataset is a pruned dataset from the original dataset from kaggle. While the original dataset contained millions of rows the current dataset contains **426880** rows and **18** columns.

The dataset contains 17 feature columns and 1 target column.


## Feature columns 

| Name   | Data Type  | Description                           |
|--------|------------|---------------------------------------|
| Id     | Integer    | A long integer identifier for a vehicle.|
| Region | Object     | A string representation of the region where the vehicle belongs to.|
| year   | Integer    | A string representation of the year of manufacture of the vehicle.|
| manufacturer   | Object    | The manifacturing entity of the vehicle.|
| model   | Object    | The model of the vehicle.|
| condition   | Object    | The current condition of the vehicle.|
| cylinders   | Object    | The number of cylinders in the vehicle.|
| fuel   | Object    | The fuel type used in the vehicle. eg: gas|
| odometer   | Float    | The current odometer reading of the vehicle in miles.|
| title_status   | Object    | What is the current title status of the vehicle? eg: clean|
| transmission   | Object    | The transmission type of the vehicle? eg: auto, manual|
| VIN   | Object    | The VIN identifier of the vehicle|
| drive   | Object    | The drive type of the vehicle eg: 4 wheel drive|
| size   | Object    | The size of the vehicle eg: full or mid size|
| type   | Object    | The type of the vehicle eg: sedan, SUV or pickup|
| paint_color   | Object    | The color of the vehicle eg: red, silver etc|
| state   | Object    | The 2 letter US state code where the vehicle is registered. eg:ca, al etc|


## Target column

The target column is **price**, and is an integer representation of the price of the vehicle in USD.

# Data Preparation


## Data Quality 

### Type correction

Some of the column types were modifed for easy handling

|Column       |  Source Type| Destination Type|
|-------------|-------------|-----------------|
| year        | float64     | int64           |
| odometer    | float64     | int64           |
| cylinders   | object      | int64           |


### Duplicate Rows
The dataset contains **0** duplicate rows 

### Missing values

|Column       |  Missing Value (Percentage)|
|-------------|----------------------------|
|year         |    0.282281                |
|manufacturer |    4.133714                |
|model        |    1.236179                |
|condition    |   40.785232                |
|cylinders    |   41.622470                |
|fuel         |    0.705819                |
|odometer     |    1.030735                |
|title_status |    1.930753                |
|transmission |    0.598763                |
|VIN          |   37.725356                |
|drive        |   30.586347                |
|size         |   71.767476                |
|type         |   21.752717                |
|paint_color  |   30.501078                |

> [!NOTE]
> **VIN is meaninless to the model and size has very large missing values(71.76%) which will render any imputation strategy meaningless. Hence both these columns will be dropped.**

> [!NOTE]
> **Other columns with large missing values are condition, paint_color,clylinders, type and drive but we will keep them**

The following imputation strategy was applied to handle missing values 

|Column       |  Imputation strategy       |
|-------------|----------------------------|
|year         | Impute with median         |
|manufacturer | Drop rows where missing    |
|model        | Drop rows where missing    |
|condition    | Create Unknown category    |
|cylinders    | hierarchical fallback with transformation and imputation with median |
|fuel         | Impute with mode           |
|odometer     | Impute with median         |
|title_status | Create Unknown category    |
|transmission | Impute with mode           |
|VIN          | Drop column                |
|drive        | Create Unknown category    |
|size         | Drop column                |
|type         | Create Unknown category    |
|paint_color  | Create Unknown category    |

> [!IMPORTANT]
> **Post imputation showed 0 missing values in all columns**
> Post imputation the dataset contains  **404026** rows and **16** columns

## Data Validation and further cleanup

1. All rows where price was zero were dropped.
2. All rows where odometer readings were zero were dropped as these are old cars.
3. Post further cleanup the dataset contains  **372269** rows and **16** columns

## Data Visualization

### Target Visualization

A histogram of the target showed a very right skew with the max value at 3.7 billion USD which is most likely an error. 

<img alt="alt_text" width="512px" src="images/price_distribution_pre_outliers.png" />

Outlier analysis was performed using the InterQuartileRange or IQR method and **6,899** outliers were
found and removed. Post outlier removal showed a more reasonable distribution of price.

<img alt="alt_text" width="512px" src="images/price_distribution_post_outliers.png" />

## Feature Visualization

### Numeric features 

1. Odometer readings 

The plot below shows the relationship between vehicle price in USD and Odometer readings.

<img alt="alt_text" width="512px" src="images/price_distribution_by_odometer.png" />

This shows an inverse relationship between Odometer readings and vehicle price. Higher the reading lower the price.

2. Year of manufacture 

The plot below shows the relationship between vehicle price in USD and year of manufacture.

<img alt="alt_text" width="512px" src="images/price_distribution_by_year.png" />

This shows the age based depreciation of assets in a market. The market seems heavily skewed towards nearer vehicles. Vehicles manufactured before the year 2000 attract very little price and shows the general trend of vehicle depreciation beyond 10-15 years of the life of the vehicle.

3. Number of cylinders 

The plot below shows the relationship between vehicle price in USD and number of cylinders.

<img alt="alt_text" width="512px" src="images/price_distribution_by_cylinders.png" />

The plot shows that the market is dominated by the 4-6 cylinders category. It shows a strong corelation between number of cylinders and the price. However the possible luxury vehicles with more than 8 cylinders have a very small market share. 

**Correlation betwen price and numeric features**

| Feature  	   | correlation Score |correlation Indicator|    Note       |
|--------------|-----------------------|---------------|---------------|
| year.        | 0.407                 | Very Strong positive correlation| Higher the manufacturing year(newer the car) higher the price.|
| cylinders | 0.241 | Strong positive correlation| vehicles with more cylinders are generally luxury ones and command a higher price.|
| odometer| -0.2 | Strong negative correlation| As vehicles add more miles , they generally selll for less in the market|

# Modeling

# Evaluation

# Deployment