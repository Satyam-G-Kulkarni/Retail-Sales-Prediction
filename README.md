# Retail-Sales-Prediction

Objective

The primary objective of the Retail Sales Prediction Project was to develop a machine learning model for forecasting sales across a set of retail stores using the Rossman dataset. This project aimed to uncover patterns, identify influential factors, and provide actionable insights for optimizing store performance. The dataset included information about stores, competition, and sales-related features.

Approach

1. Data Loading and Cleaning
   
Imported necessary libraries and loaded two datasets: 'store.csv' and 'Rossmann Stores Data.csv.'
Cleaned the data by addressing NaN and missing values, removing duplicates, and combining both datasets for further analysis.

3. Data Wrangling
   
Removed outliers from the data.
Extracted day, time, and year from the 'Date' variable.

5. Data Visualization and Hypothesis Formulation
   
Explored data using nine different types of charts to understand relationships between variables.
Formulated hypotheses based on observed patterns and trends.

7. Data Pre-processing

Applied one-hot encoding on selected categorical variables.
Removed unnecessary features and transformed the 'Sales' feature.
Removed the 'Customers' feature due to high correlation with sales and its uncontrollable nature by stores.
Scaled the data using standard scaler.

9. ML Model Implementation
    
Split the dataset into training and test subsets.
Experimented with three ML models: Linear Regression, Decision Tree, and RandomForest Regressor.
RandomForest Regressor was chosen for its sophistication and robustness to overfitting.

11. Key Findings
    
Promotions have a positive impact on sales.
Assortment type 'b' (Extra) has higher demand.
Store type 'B,' despite being fewer in number, had the highest sales average.
Slightly higher sales were observed on school holidays.
Competition distance strongly affects sales.
RandomForest Regressor outperformed other models in terms of RMSE and R-squared values.

13. Challenges and Limitations

Data Quality: Addressed challenges related to data quality through careful preprocessing.

External Factors: Recognized limitations in capturing the full impact of external factors, such as economic conditions.

Individual Store Performance: The model is generalized for the entire Rossman store chain and may not be highly accurate for individual stores.

Future Work

While the project lays a solid foundation for retail sales prediction, opportunities for further enhancement include:

Integration of external data sources to capture broader economic trends.
Continuous model refinement and iteration based on ongoing data collection.
Deployment and monitoring strategies for real-world application.

Conclusion

In conclusion, the Retail Sales Prediction Project provides actionable insights for retail store management to optimize performance. Despite challenges, the models demonstrate robust predictive capabilities, paving the way for future advancements in the field. The key findings offer valuable guidance for decision-making and strategic planning in the retail industry.
