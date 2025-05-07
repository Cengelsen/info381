from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from skrules import SkopeRules
from preprocessing import *
import matplotlib.pyplot as plt
import shap
import numpy as np

le = LabelEncoder()

def clean_data(filename):

    print("Importing data...")
    data = pd.read_csv(datapath+filename, index_col=0)

    # Separate majority and minority classes
    df_majority = data[data['is_fraud'] == 0]
    df_minority = data[data['is_fraud'] == 1]

    # Downsample majority class to match the minority class size (or a bit more if you like)
    df_majority_downsampled = df_majority.sample(n=10000, random_state=42)

    # Combine minority class with downsampled majority
    df_balanced = pd.concat([df_minority, df_majority_downsampled])

    # Shuffle the dataset
    data = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("splitting time columns...")
    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])

    #data['trans_minute'] = data['trans_date_trans_time'].dt.minute
    #data['trans_hour'] = data['trans_date_trans_time'].dt.hour
    #data['trans_day'] = data['trans_date_trans_time'].dt.day
    data['trans_month'] = data['trans_date_trans_time'].dt.month
    data['trans_year'] = data['trans_date_trans_time'].dt.year
    data['trans_dayofweek'] = data['trans_date_trans_time'].dt.dayofweek

    data["dob"] = pd.to_datetime(data["dob"])
    #data["dob_day"] = data["dob"].dt.day
    data["dob_month"] = data["dob"].dt.month
    data["dob_year"] = data["dob"].dt.year

    print("Dropping columns...")
    data = data.drop(["cc_num","long", "merch_long", "lat", 
                      "merch_lat", "unix_time", 
                      "trans_date_trans_time",
                      "first", "last", "dob","trans_num","zip"
                      ], axis=1)
    
    print("Rounding columns...")
    data[["trans_month", "trans_year", "trans_dayofweek", "dob_month", "dob_day", "city_pop"]] = data[["trans_month", "trans_year", "trans_dayofweek", "dob_year", "dob_month", "city_pop"]].round(decimals=0)

    data["amt"] = data["amt"].round(decimals=2)

    cat_data = data.select_dtypes(include=["object"])

    print(data.columns)

    print("Encoding categorical features...")
    label_encoders = {}
    inverse_mappings = {}

    for col in cat_data.columns:
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
        inverse_mappings[col] = dict(zip(le.transform(le.classes_), le.classes_))

    print("Rounding categorical columns...")
    data[["merchant", "category", "street", "city", "state", "job"]] = data[["merchant", "category", "street", "city", "state", "job"]].round(decimals=0)

    
    print(data.head())


    return data, inverse_mappings
# Load the dataset 
data, inverse_mappings = clean_data("fraud.csv")  

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


#def predict_as_proba(X):
#    preds = skope.predict(X)  # skope.predict returns 0 or 1
#    return np.vstack([1 - preds, preds]).T  # Turn into probabilities


# Pick some explanation baseline
#X_explain = X_train.sample(1000, random_state=42)
#X_explain = X_train.sample(100, random_state=42)
#X_test_sample = X_test.sample(100, random_state=42)

# Use shap.Explainer (not KernelExplainer)
#explainer = shap.KernelExplainer(predict_as_proba, X_explain)

# Compute SHAP values
#shap_values = explainer.shap_values(X_test.iloc[:100])
#X_test_sample = X_test.sample(100, random_state=42)
#shap_values = explainer.shap_values(X_test)

#shap_values = explainer.shap_values(X_test)

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



#shap.summary_plot(shap_values[1], X_test_sample, plot_type="violin", show=False)
#shap.summary_plot(shap_values[1], X_test, plot_type="violin", show=False)

#plt.savefig("shap_summary_violin2.png")
#plt.close()

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