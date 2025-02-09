# POC example for team israel poc 

# Simulated game data for Jack Flaherty in 2024 (can be replaced with real data)
game_data = {
    "Date": pd.date_range(start="2024-03-01", periods=10, freq="4D"),
    "Pitch Count": [85, 92, 76, 98, 65, 101, 88, 73, 90, 95],
    "Rest Days": [4, 3, 5, 2, 6, 3, 4, 5, 3, 2],
    "Velocity Drop (%)": [-0.5, -1.2, 0.3, -1.5, 0, -2.0, -0.8, 0.1, -1.0, -1.3],
    "Cumulative Workload (Last 7 Days)": [165, 170, 140, 180, 120, 190, 175, 140, 160, 185],
    "Cumulative Workload (Last 14 Days)": [320, 340, 290, 360, 270, 370, 345, 290, 310, 350],
    "High-Leverage Pitches": [30, 25, 10, 40, 15, 45, 35, 12, 38, 41],  # Critical situations (e.g., runners on base, late innings)
    "Fatigue Score (%)": [65, 72, 50, 80, 45, 85, 68, 48, 75, 78]  # Actual fatigue scores for evaluation
}

# Creating the DataFrame
df_flaherty = pd.DataFrame(game_data)

# Define features including High-Leverage Situations
X_flaherty = df_flaherty[["Rest Days", "Pitch Count", "Velocity Drop (%)", 
                           "Cumulative Workload (Last 7 Days)", "Cumulative Workload (Last 14 Days)", 
                           "High-Leverage Pitches"]]
y_flaherty = df_flaherty["Fatigue Score (%)"]

# Splitting dataset into training and testing sets
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_flaherty, y_flaherty, test_size=0.2, random_state=42)

# Initializing and training the Random Forest model with high-leverage inclusion
rf_model_flaherty = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_flaherty.fit(X_train_f, y_train_f)

# Making predictions on test data
y_pred_flaherty = rf_model_flaherty.predict(X_test_f)

# Evaluating model performance
mae_flaherty = mean_absolute_error(y_test_f, y_pred_flaherty)

# Displaying model predictions
model_results_flaherty = pd.DataFrame({
    "Actual Fatigue Score": y_test_f, 
    "Predicted Fatigue Score": y_pred_flaherty
})

tools.display_dataframe_to_user(name="ML Fatigue Model for Jack Flaherty", dataframe=model_results_flaherty)

# Output Mean Absolute Error
mae_flaherty
