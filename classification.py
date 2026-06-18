from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("=" * 45)
print("      AI DATA CLASSIFICATION SYSTEM")
print("=" * 45)

print("\nDataset Loaded Successfully ✔")
print("Total Samples :", len(X))
print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))

print(f"\nModel Accuracy : {accuracy * 100:.2f}%")

print("\nChoose a flower sample to classify:")
print("1. Sample A")
print("2. Sample B")
print("3. Sample C")

choice = int(input("\nEnter your choice (1-3): "))

if choice == 1:
    sample = [[5.1, 3.5, 1.4, 0.2]]

elif choice == 2:
    sample = [[6.0, 2.7, 5.1, 1.6]]

elif choice == 3:
    sample = [[6.5, 3.0, 5.8, 2.2]]

else:
    print("❌ Invalid choice!")
    exit()

prediction = model.predict(sample)

print("\nSelected Sample:", sample[0])
print("Predicted Flower:", iris.target_names[prediction][0].capitalize())

print("\nThank you for using the AI Data Classification System!")