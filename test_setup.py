# import pandas, sklearn, xgboost, lightgbm, fastapi
# print("sklearn:", sklearn.__version__)
# print("All good")

import pandas as pd

# df = pd.read_csv("data/TelcoCustomerChurn.csv")
df = pd.read_csv("data/Telco-Customer-Churn.csv")

print(df.head())
# print(df.dtypes)
# print(df["Churn"].value_counts(normalize=True))
# print(df.isna().sum())