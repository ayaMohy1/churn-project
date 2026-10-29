import pandas as pd
import numpy as np 
import sklearn 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split , StratifiedKFold, cross_val_score,cross_val_predict
# from sklearn.ensemble import RandomForstClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix 
import os
import joblib

# load data and clean
df_churn_data = pd.read_csv("data/TelcoCustomerChurn.csv")
print("????????????????????????????")
df_churn_data['ChurnLabel']= df_churn_data['ChurnLabel'].map({'Yes':1, 'No':0}) #target 0 (stayed) and 1 (churned
# df_churn_data['Under30']= df_churn_data['Under30'].map({'Yes':1, 'No':0})
# df_churn_data['Gender']= df_churn_data['Gender'].map({'Male':1, 'Female':0})
# df_churn_data['Married']= df_churn_data['Married'].map({'Yes':1, 'No':0})


# print(df_churn_data.isnull().sum())


# columns to remove
geo_col= ['ZipCode', 'Latitude', 'Longitude']
leak_col=['CustomerStatus', 'ChurnScore','CLTV', 'ChurnCategory', 'ChurnReason']
useless_col= ["CustomerID", "City", "State", "Country"]
maybe_col= ["SatisfactionScore"]
df_churn_data= df_churn_data.drop(columns=geo_col + leak_col + useless_col + maybe_col)
# print(df_churn_data.head())
# df_churn_data['ChurnCategory']= df_churn_data['ChurnReason'].map({'Competitor':1, 'Dissatisfied':2, 'Attitude':3, 'Price':4, 'Other':5})

# get correlation matrix
# corr_matrix= df_churn_data.corr()
# print(corr_matrix)

# split data into training and testing sets
x= df_churn_data.drop(columns= ['ChurnLabel'])
y= df_churn_data['ChurnLabel']


numeric_cols =x.select_dtypes(include= ['number']).columns.tolist()
categorical_cols= x.select_dtypes(exclude= ['number']).columns.tolist()

print(numeric_cols)
print(categorical_cols)



# split data into training and testing sets
x_train, x_test,y_train , y_test= train_test_split(x,y, test_size=0.2, stratify=y, random_state=42)

print("y_test",y_test)
# preprocess data

# --- Belt 1: numbers ---
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# --- Belt 2: text ---
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])


# --- combine ---
preprocessor= ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

x_ready= preprocessor.fit_transform(x_train)
print("before", x_train.shape)
print("after", x_ready.shape)



# cross validation

# 1. Join preprocessing + model into one pipeline
# model= Pipeline([
#     ("prep", preprocessor),
#     ('model',LogisticRegression(max_iter=1000,class_weight='balanced') )
# ])

# # # 2. Choose how to cut the data
# cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

# # 3. Run cross-validation
# scores= cross_val_score(model,x_train, y_train,cv=cv, scoring='roc_auc')

# 4. Show results

# print("Scores of each fold:", scores)
# print("Average:", scores.mean())


# compare 3 model to choose the best model
# count churners vs non-churners (XGBoost needs this)
neg= (y_train== 0).sum()
pos= (y_train== 1).sum()

# the 3 competitors
models={
    "LogisticRegression":LogisticRegression(max_iter= 1000, class_weight='balanced'),
    "RandomForst": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced', random_state=42),
    "xgboost": XGBClassifier(n_estimators=300, learning_rate= 0.05, max_depth=3, scale_pos_weight= neg / pos, random_state=42)
} 

cv= StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)
# test each one with the same cross-validation

# for name, clf in models.items():
#     pipe= Pipeline([("prep", preprocessor),("model", clf)])
#     scores= cross_val_score(pipe, x_train, y_train, cv= cv, scoring= "roc_auc")
#     print(f"{name:20s} means= {scores.mean():.4f} std= {scores.std():.4f}")


# build the final model
# max_depth=3: each tree can ask at most 3 questions in a row. This keeps the trees simple, so the model doesn’t memorize the training data.
# learning_rate=0.05: how big a step each new tree takes. Small steps learn slowly but more carefully.
final_model= Pipeline([
    ("prep", preprocessor),
    ("model", XGBClassifier(n_estimators=300, learning_rate=.05, max_depth=3, scale_pos_weight=neg/ pos, random_state=42))
])

# get a probability for every training customer
cv= StratifiedKFold(n_splits=5, shuffle= True, random_state= 42)
# cross_val_predict: gives you individual predictions
train_proba= cross_val_predict(final_model, x_train, y_train, cv= cv, method='predict_proba')[:, 1]

# find the best cutoff
# starting values
best_threshold = 0.5
best_f1= 0

for t in np.arange(0.1, 0.9, 0.01):
    pred= (train_proba >= t).astype(int)
    score= f1_score(y_train, pred)
    if score > best_f1:
        best_f1 = score
        best_threshold= t

print("Best cutoff:", round(best_threshold, 2))
print("F1 at that cutoff:", round(best_f1, 3))


# the final exam on the locked test set
final_model.fit(x_train, y_train)
test_proba= final_model.predict_proba(x_test)[:,1]
test_pred= (test_proba >= best_threshold).astype(int)

print("----------")
print(test_proba)
print(test_pred)


print("roc auc", round(roc_auc_score(y_test,test_proba),4))
print("classification report", classification_report(y_test, test_pred, target_names=["stayed", "churned"]))
print("confusion matrix", confusion_matrix(y_test, test_pred))


print("*******************")
print(final_model.classes_)


# 
os.makedirs("models", exist_ok=True)
artifact= {
    "pipeline": final_model,
    "threshold":float(best_threshold),
    "features":list(x_train.columns),
    "sklearn_version": sklearn.__version__,}

joblib.dump(artifact, "models/churn_model.joblib")
print("Saved! Cutoff", round(float(best_threshold), 3))
