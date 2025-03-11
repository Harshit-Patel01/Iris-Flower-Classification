# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset from local CSV file
df = pd.read_csv('iris_data.csv')

# Define your actual column names
feature_columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
target_column = 'Species'

# Check the structure of your data
print("First 5 rows of the dataset:")
print(df.head())

print("\nColumn names in the dataset:")
print(df.columns.tolist())

# Extract features and target
X = df[feature_columns].values
# Convert species names to numeric values if they're not already
if df[target_column].dtype == 'object':
    # Create a mapping of species names to numeric values
    species_mapping = {species: i for i, species in enumerate(df[target_column].unique())}
    y = df[target_column].map(species_mapping).values
    # Keep track of the original species names for labeling
    target_names = list(species_mapping.keys())
else:
    # If species is already numeric
    y = df[target_column].values
    target_names = [f"Class {i}" for i in sorted(df[target_column].unique())]

print("\nFeature names:", feature_columns)
print("Target names:", target_names)

# Basic statistics
print("\nBasic statistics:")
print(df[feature_columns].describe())

print("\nNumber of samples for each species:")
print(df[target_column].value_counts())

# Visualize the data
plt.figure(figsize=(12, 10))

# Create a pairplot
sns.pairplot(df, hue=target_column, markers=['o', 's', 'D'])
plt.suptitle("Pairplot of Iris Dataset Features", y=1.02)
plt.savefig('iris_pairplot.png')
plt.show()

# Create a correlation matrix
plt.figure(figsize=(10, 8))
correlation = df[feature_columns].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Iris Features")
plt.savefig('iris_correlation.png')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a K-Nearest Neighbors classifier
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('iris_confusion_matrix.png')
plt.show()

# Find the optimal K value
k_range = range(1, min(26, len(X_train)))  # This ensures k doesn't exceed training samples
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores = knn.score(X_test_scaled, y_test)
    k_scores.append(scores)

plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores)
plt.xlabel('Value of K')
plt.ylabel('Testing Accuracy')
plt.title('Accuracy for Different K Values')
plt.grid(True)
plt.savefig('iris_k_values.png')
plt.show()

print("\nOptimal K value:", k_range[k_scores.index(max(k_scores))])

# Create a function to predict new iris flowers
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Create a numpy array from the input
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale the data
    new_data_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction = knn.predict(new_data_scaled)
    
    # Get the species name
    species = target_names[prediction[0]]
    
    # Get the probability
    probabilities = knn.predict_proba(new_data_scaled)[0]
    confidence = probabilities[prediction[0]]
    
    return species, confidence

# Example usage
print("\nExample prediction:")
example_iris = [5.1, 3.5, 1.4, 0.2]  # Example measurements
species, confidence = predict_iris(*example_iris)
print(f"SepalLength: {example_iris[0]}, SepalWidth: {example_iris[1]}, PetalLength: {example_iris[2]}, PetalWidth: {example_iris[3]}")
print(f"Predicted species: {species} with {confidence:.2%} confidence")