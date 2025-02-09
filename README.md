ML-Enhanced Bullpen Fatigue Model for Team Israel

Overview

The ML-Enhanced Bullpen Fatigue Model is designed to predict pitcher fatigue levels based on workload, velocity trends, and high-leverage situations. This model helps optimize bullpen usage for Team Israel in the  (WBC) by providing actionable insights into reliever effectiveness and injury risk.

Features

Fatigue Score Prediction: Uses machine learning to estimate a pitcher's fatigue based on recent workload.

High-Leverage Situations Tracking: Evaluates the number of critical pitches thrown under pressure.

Velocity Drop Analysis: Identifies fatigue-related reductions in pitch velocity.

Cumulative Workload Metrics: Tracks pitches thrown over the last 7 and 14 days to assess fatigue build-up.

Rest Day Adjustment: Factors in recovery time between outings.

Machine Learning Model: Trained using Random Forest Regression, optimized for bullpen fatigue estimation.

How It Works

Data Inputs

Game Log Data:

Date of appearance

Number of pitches thrown

Rest days between outings

Pitch velocity trends

Cumulative pitch workload (last 7 and 14 days)

High-leverage pitch count

Machine Learning Pipeline:

Feature Engineering: Extracts fatigue-relevant statistics.

Model Training: Uses Random Forest Regression for fatigue score prediction.

Evaluation: Calculates Mean Absolute Error (MAE) to measure accuracy.

Model Implementation

Dependencies

Ensure you have the following Python libraries installed:

pip install pandas numpy scikit-learn matplotlib

Running the Model

Prepare the dataset:

Load game logs and pitcher performance data.

Train the Model:

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Define features and target variable
X = df[["Rest Days", "Pitch Count", "Velocity Drop (%)", "Cumulative Workload (Last 7 Days)", "Cumulative Workload (Last 14 Days)", "High-Leverage Pitches"]]
y = df["Fatigue Score (%)"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict fatigue scores
predictions = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

Future Enhancements

Real-time Data Integration: Sync with Statcast, Trackman, and Rapsodo for live fatigue tracking.

Opponent Bullpen Analysis: Identify when opposing relievers are most vulnerable.

Injury Risk Prediction: Detect potential overuse injuries based on biomechanics data.

Game Context Factors: Include travel schedule, temperature, and strike zone difficulty in fatigue analysis.

Conclusion

This model could provide Team Israel’s coaching staff with a data-driven approach to bullpen management, ensuring optimal reliever usage during the World Baseball Classic. By leveraging machine learning and advanced analytics, the team can maximize performance and gain a competitive edge against international competition.
