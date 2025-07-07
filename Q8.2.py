import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 1: Load the dataset
df = pd.read_csv("lungcancer.csv")


# Step 2: Inspect columns and initial rows
print("Columns in dataset:", df.columns.tolist())
print("First 5 rows of the dataset:")
print(df.head())


# Step 3: Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
df.dropna(subset=['Name', 'Surname'], inplace=True)


# Step 4: Preprocess text columns
if 'Name' in df.columns and 'Surname' in df.columns:
    df['Name'] = df['Name'].str.lower()
    df['Surname'] = df['Surname'].str.lower()
    df.drop(['Name', 'Surname'], axis=1, inplace=True)


# Step 5: Define features and target
target_column = 'Result'
X = df.drop(target_column, axis=1)
y = df[target_column]


# Check class distribution
print("\nClass distribution before filtering:\n", y.value_counts())


# Remove classes with fewer than 2 samples
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
df = df[df[target_column].isin(valid_classes)]


# Redefine X and y after filtering
X = df.drop(target_column, axis=1)
y = df[target_column]


print("\nClass distribution after filtering:\n", y.value_counts())


# Step 6: Split into training and test sets with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Step 7: Train the Support Vector Machine classifier
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)


# Step 8: Make predictions
y_pred = model.predict(X_test)


# Step 9: Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))