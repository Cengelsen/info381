import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

datapath = "/home/cengelsen/Dokumenter/studier/info381/kode/info381/data/"

le = LabelEncoder()

# de topp 10 viktigste featureene fra shap
# ingen syklisk enkoding
# label encoding med oversettelse i ettertid
# ingen clustering
# ingen skalering

def clean_data(filename):

    print("Importing data...")
    data = pd.read_csv(datapath+filename, index_col=0)

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


    print("Dropping columns...")
    data = data.drop(["long", "merch_long", "lat", 
                      "merch_lat", "unix_time", 
                      "trans_date_trans_time",
                      "first", "last", "dob",
                      ], axis=1)
    
    print("Rounding columns...")
    data[["trans_minute", "trans_hour", "trans_day", "trans_month", "trans_year", "trans_dayofweek", "dob_year", "dob_month", "dob_day", "city_pop", "zip", "cc_num"]] = data[["trans_minute", "trans_hour", "trans_day", "trans_month", "trans_year", "trans_dayofweek", "dob_year", "dob_month", "dob_day", "city_pop", "zip", "cc_num"]].round(decimals=0)

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
    print(data.columns)
    print(data.shape)

    return data, inverse_mappings