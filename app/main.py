import os
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="NELFUND Equity Selection API")

# Load trained model
model = joblib.load("model/nelfund_model.pkl")


@app.get("/")
def home():
    return {"message": "NELFUND ML API is running"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "approved_prediction": int(prediction),
        "approval_probability": float(probability)
    }

@app.post("/select")
def select_beneficiaries(payload: dict):
    """
    payload structure:
    {
        "school_budget": 5000000,
        "applications": [ {student1}, {student2}, ... ]
    }
    """

    school_budget = payload["school_budget"]
    applications = payload["applications"]

    df = pd.DataFrame(applications)

    # Get probabilities
    probs = model.predict_proba(df)[:, 1]
    df["approval_probability"] = probs

    # Rank students by probability (highest first)
    df = df.sort_values(by="approval_probability", ascending=False)

    selected = []
    total_spent = 0

    for _, row in df.iterrows():
        tuition = row["tuition_amount"]

        if total_spent + tuition <= school_budget:
            selected.append(row.to_dict())
            total_spent += tuition
        else:
            break

    return {
        "total_selected": len(selected),
        "total_spent": total_spent,
        "remaining_budget": school_budget - total_spent,
        "selected_students": selected
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
