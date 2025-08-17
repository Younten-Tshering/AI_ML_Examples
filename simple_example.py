"""
simple_example.py

A basic machine learning example using scikit-learn. It loads the Iris dataset,
splits it into training and test sets, trains a logistic regression classifier,
evaluates accuracy, and plots a scatter plot of two features.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train logistic regression classifier
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    # Plot a simple scatter of two features colored by true labels
    plt.figure(figsize=(8, 6))
    # Use first two features for visualization
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=40)
    plt.title("Iris Test Data (true labels)")
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.show()


if __name__ == "__main__":
    main()
