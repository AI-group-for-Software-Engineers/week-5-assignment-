# evaluation_deployment.py

from sklearn.metrics import precision_score, recall_score, confusion_matrix
import joblib

# Predictions
y_pred = model.predict(X_test)

# Evaluation
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", conf_matrix)

# Save model for deployment
joblib.dump(model, 'dropout_model.pkl')