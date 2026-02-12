# 🎓 NELFUND Equity Allocation AI System

An AI-powered, fairness-aware student loan allocation system inspired by the Nigerian Education Loan Fund (NELFUND).

This project demonstrates how Machine Learning can be applied responsibly in public-sector financial aid distribution to ensure transparency, budget control, and fairness.

---

## 🌐 Live Demo

🔹 Backend API (FastAPI + Swagger Docs)  
https://nelfund-equity-system.onrender.com/docs  

🔹 Streamlit Frontend (Eligibility Checker)  
https://nelfund-equity-system-ui.onrender.com/

---

## 🚀 Project Overview

This system simulates how a government-backed education loan scheme can:

- Score student applications using a trained ML model
- Rank applicants based on predicted eligibility probability
- Select beneficiaries within a fixed school-level budget
- Enforce configurable fairness policies
- Provide audit-ready allocation summaries

It is fully deployed to the cloud using Render.

---

## 🧠 Core Features

### ✅ 1. Machine Learning Scoring Engine
- Supervised classification model (Logistic Regression)
- Full preprocessing pipeline (scaling + encoding)
- Probability-based approval scoring

### ✅ 2. Budget-Constrained Selection
- School submits allocated budget
- Students ranked by approval probability
- Selection stops when budget is exhausted

### ✅ 3. Configurable Fairness Policies
Each school can define its own fairness constraints:

- Minimum disability ratio
- Minimum orphan ratio
- Maximum state representation cap

Example policy:

{
"min_disability_ratio": 0.10,
"min_orphan_ratio": 0.15,
"max_state_ratio": 0.40
}


### ✅ 4. Governance & Audit Layer

Each allocation response includes:

- Total applicants
- Total selected
- Budget spent
- Remaining budget
- Fairness metrics

This ensures transparency and policy accountability.

### ✅ 5. Cloud Deployment

- FastAPI backend deployed on Render
- Streamlit frontend deployed on Render
- Production-ready CORS configuration
- Health check endpoint for monitoring

---

## 🏗 Architecture
Student / School Admin
↓
Streamlit Frontend
↓
FastAPI Backend (Render)
↓
ML Pipeline (Scikit-learn)
↓
Fairness Constraint Engine
↓
Budget-Constrained Selection


---

## 🛠 Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas
- Streamlit
- Render (Cloud Deployment)
- Joblib (Model serialization)

---

## 📊 Example Use Case

A university receives ₦5,000,000 loan allocation.

1. Students apply.
2. Applications are scored using ML.
3. System ranks applicants.
4. Budget is allocated to top candidates.
5. Fairness policy ensures representation constraints are respected.
6. Audit summary is returned.

---

## ⚖ Responsible AI Considerations

This project explores:

- Bias mitigation using configurable constraints
- Transparent decision summaries
- Policy-driven AI governance
- Fair access to education financing

---

## 📌 Future Improvements

- Fairness optimization using linear programming
- Multi-tenant school authentication
- PostgreSQL integration for audit logs
- Role-based admin dashboard
- Monitoring & model drift detection

---

## 👨‍💻 Author

Samuel Yaula Dutse  
Data Scientist | AI Systems Builder  

---

## 📬 Connect

LinkedIn: https://linkedin.com/in/samdutse  
GitHub: https://github.com/SamDutse  

---

## ⭐ If you found this project insightful, feel free to star the repository.

