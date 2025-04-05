import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from DensityAwareClustering import DensityAwareClustering

datapath = "/home/cengelsen/Dokumenter/studier/info381/kode/info381/data/"

le = LabelEncoder()

def clean_data(filename):

    print("Importing data...")
    data = pd.read_csv(datapath+filename, index_col=0)

    dac = DensityAwareClustering(eps=0.5, min_samples=max(5, int(len(data) * 0.01)))

    print("splitting time columns...")
    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])

    data['trans_minute'] = data['trans_date_trans_time'].dt.minute
    data['trans_hour'] = data['trans_date_trans_time'].dt.hour
    data['trans_day'] = data['trans_date_trans_time'].dt.day
    data['trans_month'] = data['trans_date_trans_time'].dt.month
    data['trans_year'] = data['trans_date_trans_time'].dt.year
    data['trans_dayofweek'] = data['trans_date_trans_time'].dt.dayofweek

    data["dob"] = pd.to_datetime(data["dob"])
    data["dob_day"] = data["dob"].dt.day
    data["dob_month"] = data["dob"].dt.month
    data["dob_year"] = data["dob"].dt.year


    print("Handling skewing...")
    data["amt"] = np.log(data["amt"])
    data["city_pop"] = np.log(data["city_pop"])
    
    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
    data["dob_year"] = quantile_transformer.fit_transform(data["dob_year"].values.reshape(-1, 1)).flatten()


    print("Cyclically encoding features...")
    data["trans_minute_sin"] = np.sin(2*np.pi *data["trans_minute"]/60)
    data["trans_minute_cos"] = np.cos(2*np.pi *data["trans_minute"]/60)
    data["trans_hour_sin"] = np.sin(2*np.pi *data["trans_hour"]/24)
    data["trans_hour_cos"] = np.cos(2*np.pi *data["trans_hour"]/24)
    data["trans_day_sin"] = np.sin(2*np.pi *data["trans_day"]/31)
    data["trans_day_cos"] = np.cos(2*np.pi *data["trans_day"]/31)
    data["trans_month_sin"] = np.sin(2*np.pi *data["trans_month"]/12)
    data["trans_month_cos"] = np.cos(2*np.pi *data["trans_month"]/12)
    data["trans_dayofweek_sin"] = np.sin(2*np.pi *data["trans_dayofweek"]/7)
    data["trans_dayofweek_cos"] = np.cos(2*np.pi *data["trans_dayofweek"]/7)
    data["dob_day_sin"] = np.sin(2*np.pi *data["dob_day"]/31)
    data["dob_day_cos"] = np.cos(2*np.pi *data["dob_day"]/31)
    data["dob_month_sin"] = np.sin(2*np.pi *data["dob_month"]/12)
    data["dob_month_cos"] = np.cos(2*np.pi *data["dob_month"]/12)

    data, centroids = dac.find_natural_clusters(data)

    # uncomment if you want to visualize the clustering results
    #dac.visualize_clusters(data, centroids)

    data = data.drop(["trans_minute", "trans_hour", "trans_day", "trans_month", 
                      "trans_dayofweek", "dob_day", "dob_month", "long", "merch_long",
                      "lat", "merch_lat", "cc_num", "unix_time", "trans_date_trans_time",
                      "first", "last", "dob", "zip",
                      "trans_num"], axis=1)
    
    cat_data = data.select_dtypes(include=["object"])

    print("Encoding categorical features...")
    for col in cat_data.columns:
        data[col] = le.fit_transform(data[col])

    tobnormalized = ["amt", "city_pop", "trans_year",
                     "merchant", "city", "state", "category",
                     "street", "job",
                     ]

    scaler = StandardScaler()

    print("Scaling features...")
    for col in tobnormalized:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1)).flatten()
        
    return data


data = clean_data("fraud.csv")  

X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

""" model = RandomForestClassifier(
    n_estimators=100, 
    random_state=0, 
    min_samples_leaf=1,
    min_samples_split=2, 
    max_depth=None,
    max_features=0.3,
    bootstrap=True,
    n_jobs=6,
)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 125, 150, 175],
    'max_depth': [None],
    'max_features': [0.3],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'bootstrap': [True],
    'random_state': [0],
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=2,                # 5-fold cross-validation
    n_jobs=1,           # Use all CPU cores
    verbose=2,           # Print progress
    scoring='accuracy'   # Metric to optimize
)

# Run grid search (assuming X_train, y_train are defined)
grid_search.fit(X_train, y_train)

# Results
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
print(cm)

class_names = ["non-fraud", "fraud"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues)
plt.title('Random Forest Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig("confusion_matrix.png", bbox_inches='tight') """