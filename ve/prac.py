from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
# Load the dataset
iris = datasets.load_iris()
# Split the data into features and target
X = iris.data
y = iris.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 )
# Initialize the Naive Bayes classifier
clf = GaussianNB()
# Train the classifier
clf.fit(X_train, y_train)
# Make predictions
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Display classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))