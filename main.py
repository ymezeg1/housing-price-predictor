import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
import streamlit as sl
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
#
housingData=pd.read_csv('housingData.csv')
attributes=housingData.drop(['MEDV'],axis=1)
logValue=np.log(housingData['MEDV'])
logData=pd.DataFrame(logValue,columns=['MEDV'])
rmi=0
lstati=1
ptratioi=2
r=LinearRegression().fit(attributes,logData)
stats=attributes.mean().values.reshape(1,3)
fvalues=r.predict(attributes)
mean=mean_squared_error(logData,fvalues)
root=np.sqrt(mean)

def estimate(rm,lstat,ptratio):
  stats[0][rmi]=rm
  stats[0][lstati]=lstat
  stats[0][ptratioi]=ptratio

  logPredict=r.predict(stats)[0][0]
    
  return logPredict

def price_prediction(rm,lstat,ptratio):
  logPredict=estimate(rm,lstat,ptratio)
  pricePrediction=np.e**logPredict
  final=f'The value of your property is {pricePrediction:.2f} US Dollars.'
  return final

sl.header('Boston Housing Prices')
sl.subheader("User Input")

def userInput():
  rm=sl.slider('# of rooms',1,10,5)
  lstat=sl.slider('LSTAT (Education level and % of laborers in neighborhood)',1,40,20)
  ptratio=sl.slider('PTRATIO (Student to teacher ratio of school district)',10,30,20)
  input={'rm':rm,'lstat':lstat,'ptratio':ptratio}
  inputData=pd.DataFrame(input,index=[0])
  return inputData

data=userInput()
valuation=price_prediction(data.rm,data.lstat,data.ptratio)
sl.subheader('Here is your valuation!')
sl.write(valuation)

plt.figure(figsize=(8,16))

for i,num in enumerate(housingData.select_dtypes(exclude='O')):
    ax=plt.subplot(6,2,i + 1)
    sns.histplot(x=housingData[num],ec='black',color='#626262')
    plt.title('', fontsize=16)
    
plt.tight_layout()
sl.pyplot()

plt.figure(figsize=(14,8))
ylog=np.log(housingData['MEDV'])
ylog.skew()
fig=sns.distplot(ylog)
plt.title(f'Log value w/ skew {ylog.skew()}')
sl.pyplot()
