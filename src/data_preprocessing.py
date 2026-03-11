import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;

def load_split_data():
    df=pd.read_csv("../data/raw/Insurance_Policy.csv");
    # print(df.head());

    X=df[['Age','Annual_Income_LPA','Policy_Term_Years','Sum_Assured_Lakhs']]
    y=df[['Annual_Premium_Thousands']]

    # print(X.head())
    # print(y.head())

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    return X_train,X_test,y_train,y_test;
