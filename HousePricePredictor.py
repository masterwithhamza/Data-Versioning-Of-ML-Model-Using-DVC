import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting the data
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Import preprocessing tools for scaling and encoding
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer for applying different transformations to columns
from sklearn.pipeline import Pipeline  # Import Pipeline for chaining preprocessing and modeling steps
from sklearn.linear_model import LinearRegression  # Import LinearRegression for the regression model
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
from sklearn.metrics import mean_squared_error, r2_score  # Import metrics for model evaluation

# Load the dataset from a URL
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)  # Read the dataset into a pandas DataFrame

# Explore the dataset
print(df.head())  # Print the first few rows of the dataset to get an overview
print(df.info())  # Print information about the dataset, including data types and non-null counts
print(df.describe())  # Print statistical summary of numerical columns

# Define features and target variable
X = df.drop("median_house_value", axis=1)  # Drop the target variable from the feature set
y = df["median_house_value"]  # Extract the target variable

# Identify categorical and numerical features
categorical_features = ["ocean_proximity"]  # List of categorical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()  # List of numerical features

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Replace missing values with the median value of each column
    ('scaler', StandardScaler())  # Standardize numerical features by scaling to mean 0 and variance 1
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace missing values with the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Convert categorical values to one-hot encoded vectors
])

# Combine preprocessing steps for both numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),  # Apply numerical transformations
        ('cat', categorical_transformer, categorical_features)  # Apply categorical transformations
    ])

# Create a model pipeline combining preprocessing and regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),  # First step: preprocess data
    ('regressor', LinearRegression())  # Second step: apply linear regression
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% training, 20% testing

# Train the model on the training data
model.fit(X_train, y_train)  # Fit the model using the training data

# Predict on the test set
y_pred = model.predict(X_test)  # Make predictions on the test data

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Calculate the Mean Squared Error of predictions
r2 = r2_score(y_test, y_pred)  # Calculate the R^2 Score of predictions
print(f"Mean Squared Error: {mse}")  # Print the Mean Squared Error
print(f"R^2 Score: {r2}")  # Print the R^2 Score
