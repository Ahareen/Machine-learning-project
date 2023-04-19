import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

#loading the data from csv file to pandas dataframe
car_dataset=pd.read_csv('/home/adita/Desktop/car_ml/car data.csv')

#inspecting the first five rows of the dataframe
car_dataset.head()

#checking the number of rows and colums
car_dataset.shape

#getting some more information on the dataset
car_dataset.info()

#checking the number of missing values
car_dataset.isnull().sum()

#checking the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

#encoding Fuel_Type column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

#encoding Seller_Type column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

#encoding Transmission column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

car_dataset.head()

#splitting the data into training and testing data
X=car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y=car_dataset['Selling_Price']

#print(X)
#print(Y)

#Splitting the Training and Testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=2)

#model training
#loading the linear regression model
lin_reg_model=LinearRegression()

lin_reg_model.fit(X_train,Y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

#prediction on training data
training_data_prediction=lin_reg_model.predict(X_train)

#R-squared error
error_score=metrics.r2_score(Y_train,training_data_prediction)
print("R-squared error:",error_score)

#Visualise the actual prices and predicted prices
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual prices vs predicted prices")
plt.show()

#prediction on test data
test_data_prediction=lin_reg_model.predict(X_test)

#R-squared error
error_score=metrics.r2_score(Y_test,test_data_prediction)
print("R-squared error:",error_score)

#Visualise the actual prices and predicted prices
plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual prices vs predicted prices")
plt.show()

#loading the Lasso model
lass_reg_model=Lasso()

lass_reg_model.fit(X_train,Y_train)

Lasso(alpha=1.0,copy_X=True,fit_intercept=True,max_iter=1000,normalize=False,positive=False,precompute=False,random_state=None,selection='cyclic',tol=0.0001,warm_start=False)

#prediction on training data
training_data_prediction=lass_reg_model.predict(X_train)

#R-squared error
error_score=metrics.r2_score(Y_train,training_data_prediction)
print("R-squared error:",error_score)

#Visualise the actual prices and predicted prices
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual prices vs predicted prices")
plt.show()

#prediction on training data
test_data_prediction=lass_reg_model.predict(X_test)

#R-squared error
error_score=metrics.r2_score(Y_test,test_data_prediction)
print("R-squared error:",error_score)

#Visualise the actual prices and predicted prices
plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual prices vs predicted prices")
plt.show()

