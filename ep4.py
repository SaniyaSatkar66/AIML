import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("bank-full.csv", sep=';')  # semicolon delimiter

# Preview columns
print("Columns:", df.columns.tolist())

# Replace 'unknown' with NaN and fill with mode
df.replace('unknown', pd.NA, inplace=True)
for col in df.columns:
    if df[col].isna().sum() > 0:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode[0])
        else:
            df[col] = df[col].fillna('Unknown')

# Encode categorical variables
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Define features and target
features = ['age', 'job', 'education', 'balance', 'loan',
            'contact', 'previous', 'campaign', 'poutcome']
target = 'y'

# Normalize features to integers for MultinomialNB
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
df[features] = (df[features] * 100).astype(int)

# Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
