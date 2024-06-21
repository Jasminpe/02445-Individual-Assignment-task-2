import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar

# Loading data
data = pd.read_csv('HR_data.csv')

# Convert the frustration levels to binary categories
# Assume 0 for not frustrated, 1 for frustrated
data['Frustrated'] = (data['Frustrated'] > 0).astype(int)  

# Split the dataset by individuals to avoid data leakage
unique_individuals = data['Individual'].unique()
train_individuals, test_individuals = train_test_split(unique_individuals, test_size=0.3, random_state=42)

# Create training and testing sets based on the split individuals
train_data = data[data['Individual'].isin(train_individuals)]
test_data = data[data['Individual'].isin(test_individuals)]

# One-Hot Encoding categorical variables
train_data_encoded = pd.get_dummies(train_data, columns=['Round', 'Phase', 'Cohort'], drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=['Round', 'Phase', 'Cohort'], drop_first=True)

# Features and target
features = train_data_encoded.drop(['Frustrated', 'Unnamed: 0', 'Individual', 'Puzzler'], axis=1).columns
X_train = train_data_encoded[features]
y_train = train_data_encoded['Frustrated']
X_test = test_data_encoded[features]
y_test = test_data_encoded['Frustrated']

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model
log_reg = LogisticRegression(random_state=42)

# Train the model
log_reg.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Evaluate the model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
report_log_reg = classification_report(y_test, y_pred_log_reg)

print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy_log_reg * 100:.2f}%")
print("Classification Report:")
print(report_log_reg)

# Initialize the Random Forest model
rf_classifier = RandomForestClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)

# Train the model
grid_search_rf.fit(X_train_scaled, y_train)

# Best parameters
best_rf = grid_search_rf.best_estimator_

# Predict on the test set
y_pred_rf = best_rf.predict(X_test_scaled)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print(f"Best Parameters for Random Forest: {grid_search_rf.best_params_}")
print("Random Forest Classifier Performance:")
print(f"Accuracy: {accuracy_rf * 100:.2f}%")
print("Classification Report:")
print(report_rf)

# McNemar's Test
contingency_table = pd.crosstab(y_pred_log_reg, y_pred_rf)
result = mcnemar(contingency_table, exact=True)

print('Contingency Table:')
print(contingency_table)
print(f'McNemar\'s Test p-value: {result.pvalue}')

if result.pvalue < 0.05:
    print('The difference in performance between Logistic Regression and Random Forest is statistically significant.')
else:
    print('The difference in performance between Logistic Regression and Random Forest is not statistically significant.')