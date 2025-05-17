# YOUR CODE HERE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Step 1: Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 2: Create training, validation, and test datasets (60/20/20 split)
X = df[['cocoa_percent', 'maker_location', 'specific_origin']]  # Features of interest
y = df['bean_type_class']  # Target variable

# Split into training (60%), validation (20%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 3: Identify features of interest
# Features: cocoa_percent, maker_location, and specific_origin

# Step 4: Clean and Standardize Features

# Handle missing values (if any)
X_train['maker_location'] = X_train['maker_location'].fillna('Unknown')
X_val['maker_location'] = X_val['maker_location'].fillna('Unknown')
X_test['maker_location'] = X_test['maker_location'].fillna('Unknown')

X_train['specific_origin'] = X_train['specific_origin'].fillna('Unknown')
X_val['specific_origin'] = X_val['specific_origin'].fillna('Unknown')
X_test['specific_origin'] = X_test['specific_origin'].fillna('Unknown')

# One-hot encode categorical columns (maker_location and specific_origin)
X_train_encoded = pd.get_dummies(X_train, columns=['maker_location', 'specific_origin'], drop_first=True)
X_val_encoded = pd.get_dummies(X_val, columns=['maker_location', 'specific_origin'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=['maker_location', 'specific_origin'], drop_first=True)

# Standardize numerical features (cocoa_percent)
scaler = StandardScaler()
X_train_encoded['cocoa_percent'] = scaler.fit_transform(X_train_encoded[['cocoa_percent']])
X_val_encoded['cocoa_percent'] = scaler.transform(X_val_encoded[['cocoa_percent']])
X_test_encoded['cocoa_percent'] = scaler.transform(X_test_encoded[['cocoa_percent']])

# Verify the result by checking the first few rows of the datasets
print("Training Set:\n", X_train_encoded.head())
print("Validation Set:\n", X_val_encoded.head())
print("Test Set:\n", X_test_encoded.head())
