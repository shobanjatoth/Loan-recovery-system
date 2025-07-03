from flask import Flask, request, render_template
from src.pipline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Risk strategy logic
def assign_recovery_strategy(risk_score):
    if risk_score > 0.75:
        return "Immediate legal notices & aggressive recovery attempts"
    elif 0.50 <= risk_score <= 0.75:
        return "Settlement offers & repayment plans"
    else:
        return "Automated reminders & monitoring"


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        def safe_get(name, type_cast=str):
            val = request.form.get(name)
            if val is None or val.strip() == "":
                raise ValueError(f"Missing field: {name}")
            return type_cast(val)

        # Input data from form
        input_data = {
            "Age": safe_get("Age", int),
            "Gender": safe_get("Gender"),
            "Employment_Type": safe_get("Employment_Type"),
            "Monthly_Income": safe_get("Monthly_Income", float),
            "Num_Dependents": safe_get("Num_Dependents", int),
            "Loan_Amount": safe_get("Loan_Amount", float),
            "Loan_Tenure": safe_get("Loan_Tenure", int),
            "Interest_Rate": safe_get("Interest_Rate", float),
            "Collateral_Value": safe_get("Collateral_Value", float),
            "Outstanding_Loan_Amount": safe_get("Outstanding_Loan_Amount", float),
            "Monthly_EMI": safe_get("Monthly_EMI", float),
            "Payment_History": safe_get("Payment_History"),
            "Num_Missed_Payments": safe_get("Num_Missed_Payments", int),
            "Days_Past_Due": safe_get("Days_Past_Due", int),
            "Collection_Attempts": safe_get("Collection_Attempts", int),
            "Collection_Method": safe_get("Collection_Method"),
            "Legal_Action_Taken": safe_get("Legal_Action_Taken")
        }

        # Run prediction
        pipeline = PredictionPipeline()
        result = pipeline.predict(input_data)
        result["Recovery_Strategy"] = assign_recovery_strategy(result["Risk_Score"])

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == '__main__':
    app.run(debug=True)

