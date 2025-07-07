import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("Housing.csv")
print("Dataset preview:")
print(df.head())

# Step 1: Handle missing values using imputer (mean for numeric, most frequent for categorical)
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Impute numeric columns with mean
imputer_num = SimpleImputer(strategy='mean')
df[num_cols] = imputer_num.fit_transform(df[num_cols])

# Impute categorical columns with most frequent
imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# Step 2: Encode categorical variables
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 3: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 4: Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Step 5: Add cluster labels to original DataFrame
df['Cluster'] = kmeans.labels_

# Step 6: Show final result
print("\nClustered Data Preview:")
print(df.head())