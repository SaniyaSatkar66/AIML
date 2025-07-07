#q1 WINE QUALITY
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#load the dataset
filePath = "WineQuality.csv"
df = pd.read_csv(filePath)


df.head()


#drop unnecessary columns
df= df.drop('Id',axis=1)
#converting these columns to numeric bcoz they are string type as they contain values such as 'error'
df['citric acid'] = pd.to_numeric(df['citric acid'], errors='coerce')
df['sulphates'] = pd.to_numeric(df['sulphates'], errors='coerce')
df['quality'] = pd.to_numeric(df['quality'], errors='coerce')


#filling all na values with mean of the column
df.fillna(df.mean(),inplace=True)#selecting features
X= df[['citric acid']]
y = df['quality']


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#creating the model
model=LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test) # to make pred.


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("mse = ",mse,"r2=",r2)
# Plotting the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Simple Linear Regression')
plt.show()


#Q2 social ntw ads svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler ,LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# Load dataset
df = pd.read_csv('Social_Network_Ads.csv')
print(df)


# Drop 'User ID' since it's not useful
df.drop(columns='User ID', inplace=True)


# Drop 'Gender' if all values are missing
if df['Gender'].isna().all():
    df.drop(columns='Gender', inplace=True)


df["Gender"] = df["Gender"].str.strip().str.upper()
df["Gender"] = df["Gender"].replace({"MALE?": "MALE"})  # fix irregular values


# Encode Gender (MALE=1, FEMALE=0)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])    


# Drop rows with missing target
df = df[df['Purchased'].notna()]
df['Purchased'] = df['Purchased'].astype(int)


# Features and target
X = df.drop('Purchased', axis=1)
y = df['Purchased']


# Fill missing values in features (numerical only)
X = X.fillna(X.mean(numeric_only=True))


# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train SVM
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)




y_pred=classifier.predict(X_test)


print("confusion matrix:\n",confusion_matrix(y_test,y_pred))
print("classification report:\n",classification_report(y_test,y_pred))
print("accuracy score:\n",accuracy_score(y_test,y_pred))


#Q3 social ntw ads naive
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset
data = pd.read_csv("Social_Network_Ads.csv")


# 2. Encode Gender column (Male/Female to 0/1)
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])


# 3. Drop rows where target 'Purchased' is missing
data = data.dropna(subset=['Purchased'])


#  Ensure 'Purchased' is integer type
data['Purchased'] = data['Purchased'].astype(int)


# 4. Define features and target
X = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']


# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# 6. Impute missing values ONLY for features
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# 7. Scale features to [0, 1] for MultinomialNB
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# 8. Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_scaled, y_train)


# 9. Predict and Evaluate
y_pred = model.predict(X_test_scaled)


print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))




#Q4)
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




#Q5 online retail
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Load dataset
df = pd.read_csv('OnlineRetail_glaxo.csv', encoding='ISO-8859-1')


# Clean data
df.dropna(subset=['CustomerID', 'Quantity', 'UnitPrice', 'InvoiceDate'], inplace=True)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['CustomerID'] = df['CustomerID'].astype(str)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df.dropna(subset=['InvoiceDate'], inplace=True)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']


# RFM calculation
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}).reset_index()


# Normalize RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])


# Elbow method to find optimal clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.tight_layout()
plt.show()


# K-Means Clustering (choose optimal k based on elbow plot, here we assume k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)


# Cluster summary
summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Num_Customers'})


print("\nCluster Summary:\n")
print(summary)


# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(rfm_scaled)
rfm['PCA1'] = pca_result[:, 0]
rfm['PCA2'] = pca_result[:, 1]


plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('Customer Segments (RFM Clustering)')
plt.tight_layout()
plt.show()


#Q6 shoppers intentions svm
import joblib
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df= pd.read_csv('online_shoppers_intention.csv')






label_encoder={}
for col in['Month','VisitorType','Weekend']:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    label_encoder[col]=le






df.dropna(inplace=True)


X=df.drop('Revenue',axis=1)
y=df['Revenue']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.39,random_state=42)


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


svm_model=SVC(kernel='linear')
svm_model.fit(X_train,y_train)


joblib.dump(svm_model,'svm_model7.pkl')
joblib.dump(scaler,'scaler7.pkl')
print("model saved")


loaded_model=joblib.load('svm_model7.pkl')
loaded_scaler=joblib.load('scaler7.pkl')


y_pred=svm_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy score of model:{accuracy *100:.2f}")


#Q7) naive shoppers intentions
import joblib
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


df=pd.read_csv('online_shoppers_intention.csv')


def clean_revenue(val):
    if pd.isna(val):
        return np.nan
    val=str(val).strip().lower()
    if 'true'in val:
        return 1
    elif 'false' in val:
        return 0
    else:
        return np.nan


df['Revenue']=df['Revenue'].apply(clean_revenue)
df.dropna(subset='Revenue',inplace=True)
df['Revenue']=df['Revenue'].astype(int)


label_encoder={}
for col in ['Month','Weekend','VisitorType']:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    label_encoder[col]=le


df.dropna(inplace=True)




X=df.drop('Revenue',axis=1)
y=df['Revenue']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


nb_model=GaussianNB()
nb_model.fit(X_train,y_train)


joblib.dump(nb_model,'nb_model7.pkl')
joblib.dump(scaler,'7scaler.pkl')
print('model saved')


loaded_model=joblib.load('nb_model7.pkl')
loaded_scaler=joblib.load('7scaler.pkl')


y_pred=nb_model.predict(X_test)


accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy score: {accuracy*100:.2f}%")
print("\n classification report:",classification_report(y_test,y_pred))


#Q8 svm lung cancer
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df=pd.read_csv('lungcancer.csv')
print(df.head())


df.drop(columns=['Name','Surname'],inplace=True)
df.dropna(inplace=True)


X=df.drop('Result',axis=1)
y=df['Result']
y=df['Result'].astype(int)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


svm_model=SVC(kernel='linear')
svm_model.fit(X_train,y_train)


joblib.dump(scaler,'8scaler.pkl')
joblib.dump(svm_model,'svm_model8.pkl')
print('model saved')


loaded_model=joblib.load('svm_model8.pkl')
loaded_scaler=joblib.load('8scaler.pkl')


y_pred=loaded_model.predict(X_test)


accuracy=accuracy_score(y_test,y_pred)
print(f"accuracy:{accuracy*100:.2f}%")






#Q8 svm 


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





#Q9)
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




#q10) housing 
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


#Q11)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the dataset
df = pd.read_csv("Gas_turbine.csv")


# Step 2: Display the first few rows
print("First 5 rows:")
print(df.head())


# Step 3: Check and remove rows with non-numeric values
df = df.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric; invalid ones become NaN
df.dropna(inplace=True)  # Remove rows with NaN (i.e., originally non-numeric values)


# Step 4: Print cleaned column names
print("\nColumns:")
print(df.columns)


# Step 5: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


# Step 6: Apply K-Means
k = 3  # Choose appropriate number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)


# Step 7: Add cluster labels to the DataFrame
df['Cluster'] = clusters


# Step 8: Save to new CSV
df.to_csv("Gas_turbine_clustered.csv", index=False)
print("\nClustered data saved to 'Gas_turbine_clustered.csv'")


# Optional: Visualize using first two features
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['Cluster'], palette='Set1')
plt.title("Gas Turbine Clustering")
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.grid(True)
plt.show()


#Q12glaxo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load the data
df = pd.read_csv("glaxo.csv")


# Replace string 'NAN' with actual NaN
df.replace("NAN", pd.NA, inplace=True)


# Drop rows with any missing values
df.dropna(inplace=True)


# Rename columns for clarity (optional)
df.rename(columns={
    'Total Trade Quantity': 'Volume',
    'Turnover (Lacs)': 'Turnover'
}, inplace=True)


# Define features (X) and target (y)
X = df[['Open', 'High', 'Low', 'Volume', 'Turnover']]
y = df['Close']


# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual Close Price")
plt.ylabel("Predicted Close Price")
plt.title("Actual vs Predicted Closing Prices")
plt.grid(True)
plt.show()




#Q 13 Fraud detection svm    
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# Step 1: Load the dataset
df = pd.read_csv("fraud_detection.csv")  # Ensure this CSV is in the same directory as the script


# Step 2: Drop rows where target is missing
df = df.dropna(subset=['Fraudulent'])


# Step 3: Convert Transaction_Amount to numeric (may have formatting issues)
df['Transaction_Amount'] = pd.to_numeric(df['Transaction_Amount'], errors='coerce')


# Step 4: Drop non-informative columns (IDs)
df = df.drop(columns=['Transaction_ID', 'User_ID'])


# Step 5: Drop rows with any remaining missing values
df = df.dropna()


# Step 6: Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Step 7: Prepare features and target
X = df.drop(columns='Fraudulent')
y = df['Fraudulent']


# Step 8: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 9: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Step 10: Train the SVM classifier
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')  # Added class_weight
svm_model.fit(X_train, y_train)


# Step 11: Predict and evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


print(f"\nSVM Classification Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)



#Q14fraud_detect_multinomialNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
df = pd.read_csv("fraud_detection.csv")


# Drop unnecessary columns
df = df.drop(columns=["Transaction_ID", "User_ID"])


# Fill missing values
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])


# Split into features and label
X = df.drop("Fraudulent", axis=1)
y = df["Fraudulent"]


# Encode categorical features with OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)


# Ensure all values are non-negative integers (MultinomialNB requires this)
X_encoded = X_encoded.astype(int)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# Train Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




#q14) fraud det using CategoricalNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder


# Load dataset
df = pd.read_csv("fraud_detection.csv")


# Drop unnecessary columns
df = df.drop(columns=["Transaction_ID", "User_ID"])


# Fill missing values
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])


# Split into features and target
X = df.drop("Fraudulent", axis=1)
y = df["Fraudulent"]


# Encode categorical features with OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# Train the Categorical Naive Bayes model
model = CategoricalNB()
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



#Q15)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# Step 1: Load dataset
df = pd.read_csv("fraud_detection.csv")  # Ensure the CSV is in the same directory


# Step 2: Drop rows with missing target
df = df.dropna(subset=['Fraudulent'])


# Step 3: Convert Transaction_Amount to numeric (if needed)
df['Transaction_Amount'] = pd.to_numeric(df['Transaction_Amount'], errors='coerce')


# Step 4: Drop non-informative columns (IDs)
df = df.drop(columns=['Transaction_ID', 'User_ID'])


# Step 5: Drop remaining rows with missing values
df = df.dropna()


# Step 6: Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Step 7: Split into features and target
X = df.drop(columns='Fraudulent')
y = df['Fraudulent']


# Step 8: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 9: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Step 10: Train SVM classifier with linear kernel (fast)
svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)


# Step 11: Predictions and Evaluation
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# Step 12: Display Results
print(f"\nSVM Classification Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)




Q16
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


# Step 1: Load the dataset
df = pd.read_csv("forest.csv")  # Make sure this file is in the same directory as your script


# Step 2: Drop rows with missing values
df = df.dropna()


# Step 3: Encode categorical columns (e.g., month, day, size_category)
categorical_cols = df.select_dtypes(include='object').columns


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Step 4: Split into features and target
X = df.drop(columns='size_category')  # 'size_category' is the target column
y = df['size_category']


# Step 5: Standardize feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Step 7: Train the SVM classifier
svm_clf = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
svm_clf.fit(X_train, y_train)


# Step 8: Make predictions and evaluate
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# Step 9: Print the results
print(f"SVM Classification Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)




#Q20)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv('CrimeDataset.csv')
print(df.info())
print(df.isnull().sum())

df['income']=pd.to_numeric(df['income'],errors='coerce')
df['physicians']=pd.to_numeric(df['physicians'],errors='coerce')
df['hospital_beds']=pd.to_numeric(df['hospital_beds'],errors='coerce')
df=df.fillna(df.mean(numeric_only=True))

print(df.isnull().sum())

X= df.drop('crime_rate',axis=1)
y=df['crime_rate']

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.3,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("r2 score:",r2_score(y_test,y_pred))
print("mse:",mean_squared_error(y_test,y_pred))
print("mae:",mean_absolute_error(y_test,y_pred))


#Q40) bank note data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load and clean the dataset
df = pd.read_csv('bank_note_data.csv')
df = df.dropna()  # Drop rows with missing values


# Display first few rows
print("First five rows of the dataset:")
print(df.head())


# Define features and target based on actual column names
feature_columns = ['Image.Var', 'Image.Skew%', 'Image.Curt@', 'Entropy!']
X = df[feature_columns]
y = df['Class']


# Normalize features for MultinomialNB
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)


# Initialize and train the Multinomial Naive Bayes classifier
mnb = MultinomialNB()
mnb.fit(X_train, y_train)


# Predict on test set
y_pred = mnb.predict(X_test)


# Evaluate performance
print(f"\nAccuracy on test set: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Predict on a sample input
sample = [[2.3, 7.2, -1.3, 0.5]]
sample_scaled = scaler.transform(sample)
prediction = mnb.predict(sample_scaled)
print(f"\nPrediction for sample {sample}: {'Counterfeit' if prediction[0] == 1 else 'Genuine'}")





#Q39)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv("Flight_Price_Dataset_of_Bangladesh.csv")


# Show column names for verification
print("Columns in dataset:", df.columns)


# Drop rows with missing values
df = df.dropna()


# Use correct column names from dataset
features = ['Base Fare (BDT)', 'Duration (hrs)', 'Class', 'Booking Source',
            'Seasonality', 'Days Before Departure']
target = 'Total Fare (BDT)'


# Split features and target
X = df[features]
y = df[target]


# Categorical and numerical features
categorical_features = ['Class', 'Booking Source', 'Seasonality']
numerical_features = ['Base Fare (BDT)', 'Duration (hrs)', 'Days Before Departure']


# Preprocessing and model pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features)
], remainder='passthrough')


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model
pipeline.fit(X_train, y_train)


# Predict and evaluate
y_pred = pipeline.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))




#Q38)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load dataset
df = pd.read_csv('family_anxiety_14_dataset.csv')


# Drop missing values if any
df = df.dropna()


# Display first few rows
print("First five rows of the dataset:")
print(df.head())


# Rename columns for easier handling
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")


# Define target and features
target = 'anxiety_level_1-10'


# Choose relevant features from the dataset
numeric_features = ['age', 'diet_quality_1-10']
categorical_features = ['gender', 'occupation', 'recent_major_life_event']


X = df[numeric_features + categorical_features]
y = df[target]


# Preprocessing pipeline
numeric_transformer = StandardScaler()
cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)


# Choose model: Linear Regression
model = LinearRegression()


# Build pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
pipeline.fit(X_train, y_train)


# Predict
y_pred = pipeline.predict(X_test)


# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print(f"Test RMSE: {rmse:.3f}")
print(f"Test R^2: {r2:.3f}")


# Example prediction
sample = pd.DataFrame([{  
    'age': 30,
    'diet_quality_1-10': 7,
    'gender': 'female',
    'occupation': 'Student',
    'recent_major_life_event': 'No'
}])
pred_sample = pipeline.predict(sample)
print(f"Predicted anxiety level for sample: {pred_sample[0]:.1f}")



#Q37)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# Load dataset
df = pd.read_csv("yelp.csv")


# Convert and clean
df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
df['cool'] = pd.to_numeric(df['cool'], errors='coerce').fillna(0)
df['useful'] = pd.to_numeric(df['useful'], errors='coerce').fillna(0)
df['funny'] = pd.to_numeric(df['funny'], errors='coerce').fillna(0)


# Filter for binary sentiment
df = df[df['stars'].isin([1.0, 2.0, 4.0, 5.0])]
df['sentiment'] = df['stars'].apply(lambda x: 1 if x >= 4 else 0)


# Features and labels
X = df[['text', 'cool', 'useful', 'funny']]
y = df['sentiment']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('text', TfidfVectorizer(stop_words='english', max_features=5000), 'text'),
    ('numeric', StandardScaler(), ['cool', 'useful', 'funny'])
])


# Full pipeline
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', SVC(kernel='linear'))
])


# Train and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))



#Q36 SPOTIFY 2023 -NAÏVE BAYES 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import classification_report, accuracy_score 
 
# Load dataset 
df = pd.read_csv('spotify_2023.csv') 
 
# Display column names 
print("Columns in dataset:", df.columns.tolist()) 
 
# Convert 'streams' to numeric 
df['streams'] = pd.to_numeric(df['streams'], errors='coerce') 
print("After converting streams to numeric:", df['streams'].notna().sum(), "valid entries") 
 
# Drop rows where 'streams' is NaN 
df.dropna(subset=['streams'], inplace=True) 
print("After dropping NaNs from streams:", df.shape) 
 
# Create binary target variable 
median_streams = df['streams'].median() 
df['high_stream'] = (df['streams'] > median_streams).astype(int) 
 
# Features to use 
features = [ 
    'bpm', 'key', 'mode', 'danceability_%', 'energy_%', 
    'acousticness_%', 'released_year', 'released_month', 'released_day', 
    'in_spotify_playlists' 
] 
 
# Convert features to numeric 
df[features] = df[features].apply(pd.to_numeric, errors='coerce') 
 
# Check missing values 
print("Missing values per feature:\n", df[features].isna().sum()) 
 
# Handle missing values 
for col in features: 
    if df[col].isna().all(): 
        df[col].fillna(0, inplace=True)  # All values are NaN — fill with 0 
    else: 
        df[col].fillna(df[col].median(), inplace=True)  # Otherwise, fill with median 
 
# Define input and output 
X = df[features] 
y = df['high_stream'] 
 
# Final check 
print("Final shape of data:", X.shape) 
 
# Scale features 
scaler = MinMaxScaler() 
X_scaled = scaler.fit_transform(X) 
 
# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 
 
# Train model 
model = MultinomialNB() 
model.fit(X_train, y_train) 
 
# Predict 
y_pred = model.predict(X_test) 
 
# Evaluate 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("Classification Report:\n", classification_report(y_test, y_pred)) 


#Q38) 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load dataset
df = pd.read_csv('family_anxiety_14_dataset.csv')


# Drop missing values if any
df = df.dropna()


# Display first few rows
print("First five rows of the dataset:")
print(df.head())


# Rename columns for easier handling
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")


# Define target and features
target = 'anxiety_level_1-10'


# Choose relevant features from the dataset
numeric_features = ['age', 'diet_quality_1-10']
categorical_features = ['gender', 'occupation', 'recent_major_life_event']


X = df[numeric_features + categorical_features]
y = df[target]


# Preprocessing pipeline
numeric_transformer = StandardScaler()
cat_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)


# Choose model: Linear Regression
model = LinearRegression()


# Build pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
pipeline.fit(X_train, y_train)


# Predict
y_pred = pipeline.predict(X_test)


# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print(f"Test RMSE: {rmse:.3f}")
print(f"Test R^2: {r2:.3f}")


# Example prediction
sample = pd.DataFrame([{  
    'age': 30,
    'diet_quality_1-10': 7,
    'gender': 'female',
    'occupation': 'Student',
    'recent_major_life_event': 'No'
}])
pred_sample = pipeline.predict(sample)
print(f"Predicted anxiety level for sample: {pred_sample[0]:.1f}")




#Q35
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# Load the CSV
df = pd.read_csv("laptops_dataset.csv")


# Drop rows where review or rating is missing
df = df.dropna(subset=['review', 'rating'])


# Convert rating to numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')


# Filter only strong sentiments (1, 2 = negative; 4, 5 = positive)
df = df[df['rating'].isin([1, 2, 4, 5])]


# Create sentiment labels: 1 for positive, 0 for negative
df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)


# Combine review and product name into one text column
df['combined_text'] = df['product_name'].fillna('') + ' ' + df['review']


# Define features and label
X = df[['combined_text', 'rating']]
y = df['sentiment']


# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('text', TfidfVectorizer(stop_words='english', max_features=5000), 'combined_text'),
    ('numeric', StandardScaler(), ['rating'])
])


# Build the pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', SVC(kernel='linear'))
])


# Train the model
pipeline.fit(X_train, y_train)


# Predict and evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))



#Q34)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("home_price.csv")


# Rename target
df = df.rename(columns={'Fiyat': 'price'})
df = df.dropna(subset=['price'])


# Coerce numeric columns
for col in ['Net_Metrekare', 'Brüt_Metrekare', 'Binanın_Yaşı']:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Split features and target
X = df.drop('price', axis=1)
y = df['price']


# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()


print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)


# Pipelines for preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


# Full pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Evaluate
print(f"\nR² Score: {r2_score(y_test, y_pred):.2f}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")


# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Home Prices")
plt.grid(True)
plt.show()




#Q33)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("grammy_winners.csv")


# Display structure
print("Columns in dataset:", df.columns.tolist())
print("First 5 rows of dataset:")
print(df.head())


# Drop rows with missing target
df = df.dropna(subset=['winner'])


# Define features - use only existing columns
desired_features = ['category', 'artist', 'title', 'year']
features = [col for col in desired_features if col in df.columns]


print(f"Using features: {features}")


X = df[features]
y = df['winner']


# Convert target 'winner' to integer labels (0 and 1)
y = y.astype(int)


# Fill missing feature values
X = X.fillna('Unknown')


# Encode categorical features using LabelEncoder
label_encoders = {}
for col in features:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le


# Convert year to integer if it exists
if 'year' in features:
    X['year'] = pd.to_numeric(X['year'], errors='coerce').fillna(0).astype(int)


# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Initialize and train the Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)


# Predict on test set
y_pred = model.predict(X_test)


# Evaluate results
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




#Q32)

