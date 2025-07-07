import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('lungcancer.csv')

# Preprocessing
# Drop identifier columns and rows with missing values
df = df.drop(['Name', 'Surname'], axis=1).dropna()

# Features and target
X = df.drop('Result', axis=1)

# Binarize the target: any non-zero result indicates cancer
y = (df['Result'] > 0).astype(int)

# Discretize continuous features by rounding to nearest integer
X = X.round().astype(int)

# Split into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['No Cancer', 'Cancer'])
cm = confusion_matrix(y_test, y_pred)

# Display results
print(f'Accuracy: {accuracy:.2f}\n')
print('Classification Report:')
print(report)
print('\nConfusion Matrix:')
cm_df = pd.DataFrame(cm, index=['Actual No','Actual Yes'], columns=['Pred No','Pred Yes'])
print(cm_df)
