import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("C:/Users/priya_vk383uf/Downloads/adult.csv")

# Encode categorical columns
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    if column != "income":  # exclude target column from encoding here
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

# Encode target column
target_le = LabelEncoder()
data["income"] = target_le.fit_transform(data["income"])

# Save label encoder if needed later
joblib.dump(target_le, "target_encoder.pkl")

# Prepare features and labels
X = data.drop("income", axis=1)
y = data["income"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "best_model.pkl")
print("✅ Model saved as best_model.pkl")
