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

    school_budget = payload["school_budget"]
    applications = payload["applications"]

    df = pd.DataFrame(applications)

    # Score applicants
    df["approval_probability"] = model.predict_proba(df)[:, 1]

    # Sort by probability
    df = df.sort_values(by="approval_probability", ascending=False)

    selected = []
    total_spent = 0

    # Fairness counters
    disability_count = 0
    orphan_count = 0
    state_counts = {}

    for _, row in df.iterrows():
        tuition = row["tuition_amount"]
        state = row["state_of_origin"]

        if total_spent + tuition > school_budget:
            continue

        # State cap: max 40%
        current_total = len(selected) if len(selected) > 0 else 1
        if state_counts.get(state, 0) / current_total > 0.4:
            continue

        selected.append(row.to_dict())
        total_spent += tuition

        # Update fairness counters
        if row["disability_status"] == 1:
            disability_count += 1

        if row["orphan_status"] == 1:
            orphan_count += 1

        state_counts[state] = state_counts.get(state, 0) + 1

    # Post-selection fairness check
    total_selected = len(selected)

    disability_ratio = disability_count / total_selected if total_selected else 0
    orphan_ratio = orphan_count / total_selected if total_selected else 0

    return {
        "total_selected": total_selected,
        "total_spent": total_spent,
        "remaining_budget": school_budget - total_spent,
        "disability_ratio": disability_ratio,
        "orphan_ratio": orphan_ratio,
        "selected_students": selected
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
