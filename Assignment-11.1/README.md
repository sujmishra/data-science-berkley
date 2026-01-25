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

The dataset contains 17 feature columns and one target column.

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







# Data Preparation

# Modeling

# Evaluation

# Deployment