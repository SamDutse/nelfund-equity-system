import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data/nelfund_synthetic_student_applications.csv")

# Create synthetic approval logic
df["approved"] = (
    (df["household_income_monthly"] < 150000) &
    (df["cgpa"] >= 2.5) &
    (df["academic_standing"] == "Good") &
    (df["has_other_scholarship"] == 0)
).astype(int)

# Drop ID column
df = df.drop(columns=["application_id"])

target = "approved"
X = df.drop(columns=[target])
y = df[target]

# Separate feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))

# Save model
joblib.dump(pipeline, "model/nelfund_model.pkl")

print("Model trained and saved successfully.")
