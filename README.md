# Boston Housing Median Home Value Prediction

## Project Overview
This project aims to build a machine learning model that predicts the median home value in different areas of Boston based on various features such as crime rate, nitric oxide concentration, proximity to highways, and more. By using these features, the model can provide insights into how different factors influence housing prices in the Boston area.

## Dataset
[Boston Housing Dataset](https://www.kaggle.com/datasets/krupadharamshi/bostonhousing/data)

### Features:
- **crime_rate**: Per capita crime rate by town.
- **zoned_land**: Proportion of residential land zoned for large lots.
- **industrial_acres**: Proportion of non-retail business acres per town.
- **river_proximity**: Whether the area bounds the Charles River (1 if yes, 0 otherwise).
- **nitric_oxides**: Nitric oxides concentration (parts per 10 million).
- **rooms_per_dwelling**: Average number of rooms per dwelling.
- **age_of_home**: Proportion of owner-occupied units built before 1940.
- **employment_distance**: Weighted distances to five Boston employment centers.
- **highway_accessibility**: Index of accessibility to radial highways.
- **property_tax_rate**: Full-value property-tax rate per $10,000.
- **pupil_teacher_ratio**: Pupil-teacher ratio by town.
- **black_residents_rate**: Proportion of Black residents by town.
- **lower_status_population**: Proportion of the population considered lower status.
- **median_home_value**: The target variable representing the median value of homes in $1000s.

## Model
For this project, the following machine learning models were considered:
- **Linear Regression**
- **Decision Trees**
- **Random Forest**

Each model's performance was evaluated based on:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (R_square)

Hyperparameters were fine-tuned to optimize prediction accuracy using GridSearchCV.

## Data Preprocessing
Outliers were detected from the dataset but not removed due to the small dataset size and for keeping things simple for the sake of this notebook.

## Results
The final model yielded the following results:
- **Best model**: Random Forest Regressor
- **Best parameters**: `(max_depth=100, criterion='poisson', n_estimators=125)`
- **Best Mean Absolute Error (MAE)**: 2.010
- **Best R² Score (r_square)**: 0.897

## Key Features
The top 2 features that significantly influenced home values were:
1. **Lower status population**
2. **Rooms per dwelling**

These features had the highest feature importance with respect to the target variable, providing insights into factors that most strongly affect Boston home prices.

## Installation and Requirements
To run the project locally, you will need to have the following Python libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Future Improvements
- Experiment with more advanced models, such as deep learning (ANN).
- Remove or smooth outliers to improve model performance.
- Explore correlation among features and other feature engineering techniques.
