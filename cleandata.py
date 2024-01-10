import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def clean_and_preprocess_data():
   
    features = pd.read_csv("NUSW-NB15_features.csv", encoding='ISO-8859-1')
    data = pd.read_csv("UNSW-NB15_4.csv", header=None)
    data.columns = features['Name'].tolist()
    print("Columns in the dataset before processing:")
    print(data.head())

    #drop columns
    columns_to_drop = ['dur', 'Sload', 'Dload', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'Label']
    data = data.drop(columns=columns_to_drop, axis=1)

    data = data.apply(lambda col: pd.Categorical(col).codes if col.dtype == 'object' else col)

    # impute columns
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # when compelete
    print("\nColumns in the dataset after processing:")
    print(data.head())

    # cleaned dataset
    data.to_csv("lab8official.csv", index=False)

if __name__ == "__main__":
    clean_and_preprocess_data()
