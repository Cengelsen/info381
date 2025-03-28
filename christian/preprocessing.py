import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import yeojohnson
from sklearn.preprocessing import QuantileTransformer

from DensityAwareClustering import DensityAwareClustering
from GPUDensityCalculation import ScalableDistanceClustering

datapath = "/home/cengelsen/Dokumenter/studier/info381/kode/info381/data/"

le = LabelEncoder()
""" 
def cluster_geolocations(df, n_clusters=10, random_state=42):

    scaler = StandardScaler()

    # Prepare the data
    X = df[['lat', 'long', 'merch_long', 'merch_lat']].values
    X_scaled = scaler.fit_transform(X)
    
    # Perform MiniBatchKMeans clustering
    minibatch_kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, 
        random_state=random_state,
        batch_size=1024,  # Adjust based on memory
        max_iter=100,     # Limit iterations
        n_init=5          # Fewer initializations
    )
    
    print(f"Performing clustering with {n_clusters} clusters...")

    # Fit and predict clusters
    clusters = minibatch_kmeans.fit_predict(X_scaled)
    
    print(f"Clustering complete. Found {len(np.unique(clusters))} clusters.")

    # Create result DataFrame
    df_clustered = df.copy()
    df_clustered['cluster_id'] = clusters
        
    return df_clustered
"""

def clean_data(filename):
    data = pd.read_csv(datapath+filename, index_col=0)
    dac = DensityAwareClustering(eps=0.5, min_samples=max(5, int(len(data) * 0.01)))

    print("data imported...")
    # Data cleaning

    # Feature extraction/creation
    ## splitting time columns
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

    print("time data converted...")

    # Feature selection
    ## Removing irrelevant columns
    data = data.drop(["cc_num", "unix_time", "trans_date_trans_time",
                      "first", "last", "street", "dob", "job", "zip",
                      "trans_num"], axis=1)

    # Feature scaling/normalization

    """
    Handle skewing:
    - amt - positive skew - log transformation
    - long - negative skew - yeo-johnson
    - city_pop - positive skew - log transformation
    - merch_long - negative skew - yeo-johnson 
    - dob - slight negative skew - quantile transformation

    """

    data["amt"] = np.log(data["amt"])
    data["city_pop"] = np.log(data["city_pop"])

    #data["long"], _ = yeojohnson(data["long"])
    #data["merch_long"], _ = yeojohnson(data["merch_long"])

    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)

    data["dob_year"] = quantile_transformer.fit_transform(data["dob_year"].values.reshape(-1, 1)).flatten()

    print("skewing handled...")

    """ 
    kolonne og hvilken normalisering som er nyttigst:
    
    Cyclical encoding:
    - trans_minute
    - trans_hour
    - trans_day
    - trans_month
    - trans_dayofweek
    - dob_day
    - dob_month
    - long
    - merch_long

    """

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
    #data["long_sin"] = np.sin(2*np.pi *data["long"]/180)
    #data["long_cos"] = np.cos(2*np.pi *data["long"]/180)
    #data["merch_long_sin"] = np.sin(2*np.pi *data["merch_long"]/180)
    #data["merch_long_cos"] = np.cos(2*np.pi *data["merch_long"]/180)

    print("sin/cos transformed...")

    # Before your clustering operation
    data, init_centroids = dac.find_natural_clusters(data)

    dac.visualize_clusters(data, init_centroids)

    print("clustering done...")

    data = data.drop(["trans_minute", "trans_hour", "trans_day", "trans_month", 
                      "trans_dayofweek", "dob_day", "dob_month", "long", "merch_long",
                      "lat", "merch_lat"], axis=1)

    print("columns dropped...")
    
    cat_data = data.select_dtypes(include=["object"])

    for col in cat_data.columns:
        data[col] = le.fit_transform(data[col])

    print("categorical data encoded...")

    print(data.dtypes)

    print(data.head())

    return data


testdata = clean_data("fraudTest.csv")
traindata = clean_data("fraudTrain.csv")

""" sns.set_theme(style="whitegrid")

num_features = len(traindata.columns)
ncols = 3
nrows = int(np.ceil(num_features/ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4*nrows))

axes = axes.flatten()

for ax in axes[len(traindata.columns):]:
    ax.axis('off')

for i, col in enumerate(traindata.columns):

    print(col)

    if i < num_features:  # Ensure we don't go out of bounds
        sns.histplot(traindata[col], kde=True, ax=axes[i])
        #axes[i].set_title(col, fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

for i in range(num_features, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("confusion_mlp_tfidf.png", bbox_inches='tight')

 """
X_train = traindata.drop("is_fraud", axis=1)
X_test = testdata.drop("is_fraud", axis=1)
y_train = traindata["is_fraud"]
y_test = testdata["is_fraud"]

model = RandomForestClassifier(
    n_estimators=100, 
    random_state=0, 
    min_samples_leaf=1, 
    max_depth=5,
    max_features=None,
    n_jobs=6,
)

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
plt.savefig("confusion_matrix.png", bbox_inches='tight')