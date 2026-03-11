import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

X_train_Scaled=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Projects\Insurance_Prediction\data\processed\X_train_Scaled.csv")
X_test_Scaled=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Projects\Insurance_Prediction\data\processed\X_test_Scaled.csv")
y_train=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Projects\Insurance_Prediction\data\processed\y_train.csv")
y_test=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Projects\Insurance_Prediction\data\processed\y_test.csv")

model=LinearRegression()
model.fit(X_train_Scaled,y_train)

with open(r"C:\Users\DELL\OneDrive\Desktop\Projects\Insurance_Prediction\artifacts\model.pkl",'wb') as f:
    pickle.dump(model,f)
