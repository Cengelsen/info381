from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from skrules import SkopeRules
from preprocessing import *
import matplotlib.pyplot as plt
import shap
import numpy as np

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



# Eksempel: Bruk 100 data points som forklaringsgrunnlag
#X_explain = X_train.sample(100, random_state=42)

# Bruk SHAP KernelExplainer siden SkopeRules ikke støttes direkte av TreeExplainer
#explainer = shap.KernelExplainer(lambda x: skope.predict(x).astype(float), X_explain)

# Forklar et lite utvalg (ellers blir det tregt)
#shap_values = explainer.shap_values(X_test.iloc[:10])

# Visualisering (valgfritt)
#shap.summary_plot(shap_values, X_test.iloc[:10], show=False)
#plt.savefig("shap_summary_plot.png")


def predict_as_proba(X):
    preds = skope.predict(X)  # skope.predict returns 0 or 1
    return np.vstack([1 - preds, preds]).T  # Turn into probabilities


# Pick some explanation baseline
#X_explain = X_train.sample(1000, random_state=42)
X_explain = X_train.sample(100, random_state=42)
#X_test_sample = X_test.sample(100, random_state=42)

# Use shap.Explainer (not KernelExplainer)
explainer = shap.KernelExplainer(predict_as_proba, X_explain)

# Compute SHAP values
#shap_values = explainer.shap_values(X_test.iloc[:100])
X_test_sample = X_test.sample(100, random_state=42)
#shap_values = explainer.shap_values(X_test)

shap_values = explainer.shap_values(X_test)

# Plot
#shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
#plt.savefig("shap_summary.png")
#plt.close()

#from sklearn.ensemble import GradientBoostingClassifier
#import shap

# Train a surrogate model
#surrogate_model = GradientBoostingClassifier()
#surrogate_model.fit(X_train, y_train)

# Use TreeExplainer on the surrogate model
#explainer = shap.TreeExplainer(surrogate_model)
#shap_values = explainer.shap_values(X_test_sample, check_additivity=False)



shap.summary_plot(shap_values[1], X_test_sample, plot_type="violin", show=False)
plt.savefig("shap_summary_violin2.png")
plt.close()

#plt.savefig("shap_summary_plot2.png")


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