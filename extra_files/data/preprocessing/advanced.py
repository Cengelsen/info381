import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

datapath = "/home/cengelsen/Dokumenter/studier/info381/kode/info381/data/"

le = LabelEncoder()

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
    for col in cat_data.columns:
        data[col] = le.fit_transform(data[col])

    print("Rounding categorical columns...")
    data[["merchant", "category", "street", "city", "state", "job"]] = data[["merchant", "category", "street", "city", "state", "job"]].round(decimals=0)

    print(data.head())

    return data