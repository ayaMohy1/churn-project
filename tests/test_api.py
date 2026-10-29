from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_CUSTOMER = {
    "Gender": "Male", "Age": 78, "Under30": "No", "SeniorCitizen": "Yes",
    "Married": "No", "Dependents": "No", "NumberofDependents": 0,
    "Population": 68701, "Quarter": "Q3", "ReferredaFriend": "No",
    "Number_of_Referrals": 0, "TenureinMonths": 1, "Offer": None,
    "PhoneService": "No", "AvgMonthlyLongDistanceCharges": 0,
    "MultipleLines": "No", "InternetService": "Yes", "InternetType": "DSL",
    "AvgMonthlyGBDownload": 8, "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtectionPlan": "Yes", "PremiumTechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "Yes", "StreamingMusic": "No",
    "UnlimitedData": "No", "Contract": "Month-to-Month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Bank Withdrawal",
    "MonthlyCharge": 39.65, "TotalCharges": 39.65, "TotalRefunds": 0,
    "TotalExtraDataCharges": 20, "TotalLongDistanceCharges": 0,
    "TotalRevenue": 59.65,
}


# pytest runs every function whose name starts with test_
def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_valid_customer():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert isinstance(body["will_churn"], bool)


def test_high_risk_customer_is_flagged():
    # 1 month tenure, month-to-month contract: clearly risky
    body = client.post("/predict", json=VALID_CUSTOMER).json()
    assert body["churn_probability"] > 0.7
    assert body["will_churn"] is True

# copies the valid customer and replaces one value with a wrong one. We expect status 422 (invalid input), which proves Pydantic protects the model.
def test_bad_category_is_rejected():
    bad = {**VALID_CUSTOMER, "Contract": "Monthly"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422

# copies the customer without the Age field. We expect 422 again.
def test_missing_field_is_rejected():
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "Age"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_optional_field_can_be_omitted():
    ok = {k: v for k, v in VALID_CUSTOMER.items() if k != "Offer"}
    response = client.post("/predict", json=ok)
    assert response.status_code == 200