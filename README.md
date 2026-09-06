Movie Revenue Prediction — Linear Regression Model
Overview
This project extends an earlier exploratory correlation analysis into a trained machine learning model. Instead of just measuring the correlation between a movie's production budget and its gross revenue, this model learns from historical data and predicts revenue for unseen movies.

Dataset
Source: Movie Industry dataset by Daniel Grijalva (Kaggle)
Size: ~6,800 movies, 1986–2016
Fields used: budget, gross
What this project does
Loads and cleans the raw dataset (drops missing/zero budget-gross rows)
Splits data into training (80%) and testing (20%) sets
Trains a Linear Regression model to predict gross from budget
Evaluates model performance using R² score and Mean Absolute Error
Visualizes actual vs. predicted revenue on a scatter plot
Tech Stack
Python
pandas
scikit-learn
matplotlib
How to run
Download movies.csv from the Kaggle dataset link
Place it in the same folder as movie_revenue_ml_model.py
Install dependencies:
   pip install pandas scikit-learn matplotlib
Run the script:
   python movie_revenue_ml_model.py
Check the console for model metrics (R², MAE) and view revenue_prediction_plot.png for the visualization
Key Takeaway
Confirms the strong positive relationship between production budget and gross revenue found in the earlier correlation analysis — and turns that insight into a working predictive model.

Related Work
This builds on the original EDA project in this repo, which used correlation heatmaps and scatter plots to identify budget as the strongest predictor of movie revenue.


