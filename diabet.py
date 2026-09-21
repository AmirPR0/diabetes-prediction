import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# 1. Load Dataset
# Load the diabetes dataset from the CSV file.
# The CSV file should be in the same directory as this Python file.
data = pd.read_csv("diabetes.csv")

# Display the project title.
print("=" * 60)
print("DIABETES PREDICTION PROJECT")
print("=" * 60)


# 2. Basic Data Inspection
# Check the number of rows and columns in the dataset.
print("\nDataset shape:", data.shape)

# Display the first five rows of the dataset.
print("\nFirst 5 rows:")
print(data.head())

# Check whether the dataset contains missing values.
print("\nMissing values:")
print(data.isnull().sum())


# 3. Separate Features and Target
# Separate input features from the target column.
# Outcome: 0 = No Diabetes, 1 = Diabetes
x = data.drop("Outcome", axis=1)
y = data["Outcome"]


# 4. Train / Test Split
# Split the dataset into training and testing data.
# 80% is used for training and 20% for testing.
# stratify keeps the class distribution approximately balanced.
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the number of training and testing samples.
print("\nTraining samples:", len(x_train))
print("Testing samples:", len(x_test))


# 5. Train Random Forest Model
# Create the Random Forest classifier.
random_forest = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=42
)

# Train the model using the training data.
random_forest.fit(x_train, y_train)

# Predict diabetes outcomes for the test data.
rf_predictions = random_forest.predict(x_test)

# Calculate the model accuracy.
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Create the confusion matrix for Random Forest.
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)

# Display Random Forest results.
print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"\nAccuracy: {rf_accuracy * 100:.2f}%")

# Display precision, recall and F1-score.
print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_predictions,
    target_names=["No Diabetes", "Diabetes"]
))

# Display the confusion matrix.
print("Confusion Matrix:")
print(rf_confusion_matrix)


# 6. Train Logistic Regression Model
# Create Logistic Regression as a second model for comparison.
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train the Logistic Regression model.
logistic_model.fit(x_train, y_train)

# Predict outcomes using the Logistic Regression model.
logistic_predictions = logistic_model.predict(x_test)

# Calculate Logistic Regression accuracy.
logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

# Create the confusion matrix for Logistic Regression.
logistic_confusion_matrix = confusion_matrix(
    y_test,
    logistic_predictions
)

# Display Logistic Regression results.
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"\nAccuracy: {logistic_accuracy * 100:.2f}%")

# Display precision, recall and F1-score.
print("\nClassification Report:")
print(classification_report(
    y_test,
    logistic_predictions,
    target_names=["No Diabetes", "Diabetes"]
))

# Display the confusion matrix.
print("Confusion Matrix:")
print(logistic_confusion_matrix)


# 7. Compare Models
# Compare the accuracy of both trained models.
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Random Forest: {rf_accuracy * 100:.2f}%")
print(f"Logistic Regression: {logistic_accuracy * 100:.2f}%")


# 8. Feature Importance
# Get the importance of each feature from the Random Forest model.
# Higher values indicate a greater contribution to the model.
feature_importance = pd.Series(
    random_forest.feature_importances_,
    index=x.columns
).sort_values(ascending=False)

# Display feature importance values.
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

for feature, importance in feature_importance.items():
    print(f"{feature:<25} {importance:.4f}")


# 9. Example Predictions
# Create a copy of the test data for displaying predictions.
results = x_test.copy()

# Add the actual target values.
results["Actual"] = y_test.values

# Add the predictions made by Random Forest.
results["Predicted"] = rf_predictions

# Check whether each prediction was correct.
results["Correct"] = results["Actual"] == results["Predicted"]

# Display the first 10 prediction results.
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(results.head(10).reset_index(drop=True))


# 10. Feature Importance Visualization
# Create a horizontal bar chart for feature importance.
plt.figure(figsize=(10, 6))

feature_importance.sort_values().plot(kind="barh")

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

# Adjust the layout before displaying the chart.
plt.tight_layout()
plt.show()


# 11. Confusion Matrix Visualization
# Create a simple visual representation of the Random Forest confusion matrix.
plt.figure(figsize=(6, 5))

plt.imshow(
    rf_confusion_matrix,
    interpolation="nearest"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Set labels for the two classes.
plt.xticks(
    [0, 1],
    ["No Diabetes", "Diabetes"]
)

plt.yticks(
    [0, 1],
    ["No Diabetes", "Diabetes"]
)


# Display the actual values inside the confusion matrix.
for row in range(2):
    for column in range(2):
        plt.text(
            column,
            row,
            rf_confusion_matrix[row, column],
            ha="center",
            va="center"
        )

# Adjust the layout and display the chart.
plt.tight_layout()
plt.show()


# 12. Project Disclaimer
# This project is for educational purposes only.
print("\n" + "=" * 60)
print("DISCLAIMER")
print("=" * 60)

print("This project is an educational machine learning project.")
print("It is not intended for medical diagnosis or clinical use.")

# Confirm that the program has finished successfully.
print("\nProject completed successfully.")