import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

df = pd.read_csv('Churn_Modelling.csv')
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
df = df.astype(int)

X = df.drop('Exited', axis=1)
y = df['Exited']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred)*100)
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob)*100)

y_test_reset = y_test.reset_index(drop=True)
y_pred_reset = pd.Series(y_pred).reset_index(drop=True)
acc = accuracy_score(y_test_reset, y_pred_reset)

plt.figure(figsize=(12, 4))
plt.plot(y_test_reset[:50], label='Actual', color='blue', linewidth=2)
plt.plot(y_pred_reset[:50], label='Predicted', color='red', linestyle='--', linewidth=2)
plt.text(1, 1.05, f'Accuracy: {acc*100:.2f}%', fontsize=12, color='green')
plt.title('Line Chart: Actual vs Predicted Churn (XGBoost)')
plt.xlabel('Sample Index')
plt.ylabel('Churn (0 = No, 1 = Yes)')
plt.legend()
plt.tight_layout()
plt.show()
