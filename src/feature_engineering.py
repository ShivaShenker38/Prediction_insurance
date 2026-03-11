from data_preprocessing import load_split_data;
from sklearn.preprocessing import StandardScaler;
import pandas as pd
import pickle

X_train,X_test,y_train,y_test=load_split_data();

scaler=StandardScaler()

X_train_Scaled=scaler.fit_transform(X_train)
X_test_Scaled=scaler.transform(X_test)


pd.DataFrame(X_train_Scaled).to_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\Projects\\Insurance_Prediction\\data\\processed\\X_train_Scaled.csv",index=False)
pd.DataFrame(X_test_Scaled).to_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\Projects\\Insurance_Prediction\\data\\processed\\X_test_Scaled.csv",index=False)
pd.DataFrame(y_train).to_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\Projects\\Insurance_Prediction\\data\\processed\\y_train.csv",index=False)
pd.DataFrame(y_test).to_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\Projects\\Insurance_Prediction\\data\\processed\\y_test.csv",index=False)

with open(r"C:\Users\DELL\OneDrive\Desktop\Projects\Insurance_Prediction\artifacts\scaler.pkl",'wb') as f:
    pickle.dump(scaler,f)
