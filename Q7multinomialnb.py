import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
df = pd.read_csv("online_shoppers_intention.csv")  # Ensure file is in the same directory

# Step 2: Drop rows with missing values
df = df.dropna()

# Step 3: Encode categorical variables
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Separate features and target
X = df.drop(columns='Revenue')  # 'Revenue' is the target column (True/False)
y = df['Revenue']

# Step 5: Scale features to non-negative (required for MultinomialNB)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train Multinomial Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Step 8: Predict and evaluate
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Step 9: Print results
print(f"Naive Bayes Classification Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)
