from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from skrules import SkopeRules
from preprocessing import *
import matplotlib.pyplot as plt

# Load the dataset 

data = clean_data("fraud.csv")

# Splitting features and target variable
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Step 3: Fit the SkopeRules model

skope = SkopeRules(
    feature_names=X_train.columns,
    precision_min=0.5,
    recall_min=0.01,
    n_estimators=30,
    random_state=42
)

print("fitting the model")
skope.fit(X_train, y_train)

print("finding rules for is_fraud")
rules = skope.rules_[:10]

print("printing rules")
for rule in rules:
    print(rule)
    print()
    print(20*'=')
    print()

# Step 4: Predict and evaluate
y_pred_skope = skope.predict(X_test)

""" y_score = skope.score_top_rules(X_test)

precision, recall, _ = precision_recall_curve(y_test, y_score)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve')
plt.show() """

print("SkopeRules Classification Report:")
print(classification_report(y_test, y_pred_skope))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_skope)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap="Oranges")