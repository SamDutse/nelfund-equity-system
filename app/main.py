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

    # Default fairness policy
    fairness_policy = payload.get("fairness_policy", {
        "min_disability_ratio": 0.1,
        "min_orphan_ratio": 0.15,
        "max_state_ratio": 0.4
    })

    min_disability_ratio = fairness_policy["min_disability_ratio"]
    min_orphan_ratio = fairness_policy["min_orphan_ratio"]
    max_state_ratio = fairness_policy["max_state_ratio"]

    df = pd.DataFrame(applications)

    # Score applicants
    df["approval_probability"] = model.predict_proba(df)[:, 1]

    # Sort by probability
    df = df.sort_values(by="approval_probability", ascending=False)

    selected = []
    total_spent = 0

    disability_count = 0
    orphan_count = 0
    state_counts = {}

    for _, row in df.iterrows():
        tuition = row["tuition_amount"]
        state = row["state_of_origin"]

        if total_spent + tuition > school_budget:
            continue

        current_total = len(selected) if len(selected) > 0 else 1

        # State cap enforcement
        if state_counts.get(state, 0) / current_total > max_state_ratio:
            continue

        selected.append(row.to_dict())
        total_spent += tuition

        # Update counters
        if row["disability_status"] == 1:
            disability_count += 1

        if row["orphan_status"] == 1:
            orphan_count += 1

        state_counts[state] = state_counts.get(state, 0) + 1

    total_selected = len(selected)

    disability_ratio = disability_count / total_selected if total_selected else 0
    orphan_ratio = orphan_count / total_selected if total_selected else 0

    return {
        "total_selected": total_selected,
        "total_spent": total_spent,
        "remaining_budget": school_budget - total_spent,
        "fairness_policy_used": fairness_policy,
        "fairness_metrics": {
            "disability_ratio": disability_ratio,
            "orphan_ratio": orphan_ratio
        },
        "selected_students": selected
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
