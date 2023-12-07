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

housingData=pd.read_csv('housingData.csv')
attributes=housingData.drop(['MEDV'],axis=1)
logValue=np.log(housingData['MEDV'])
logData=pd.DataFrame(logValue,comlumns=['MEDV'])
rmi=0
lstati=1
ptratioi=2
r=LinearRegression().fit(attributes,logData)
stats=attributes.mean().values.reshape(1,3)
fvalues=r.predict(attributes)
mean=mean_squared_error(logData,fvalues)
root=np.sqrt(mean)

def estimate(rm,lstat,ptratio,confidence):
  stats[0][rmi]=rm
  stats[0][lstati]=lstat
  stats[0][ptratioi]=ptratio

  logPredict=r.predict(stats)[0][0]
  if confidence.any():
    upper=logPredict+2*root
    lower=logPredict-2*root
    interval=95
  else:
    upper=logPredict+root
    lower=logPredict-root
    interval=68

  print(interval)
  return upper,lower,logPrediction,interval

def price_prediction(rm,lstat,ptratio,confidence):
  logPrediction,upper,lower,confidence=estimate(rm,lstat,ptratio,confidence)
  pricePrediction=np.around(np.e**logPrediction/10,-3)
  priceUpper=np.around(np.e**upper/10,-3)
  priceLower=np.around(np.e**lower/10,-3)
  final=f'Property value is {pricePrediction}. At {confidence}% the range is ${priceLower} to ${priceUpper}.'
  return final

container=sl.container(border=True)
container.write("User Input")
container.sl.slider('RM',1,10,5)
container.sl.slider('LSTAT',1,40,20)
container.sl.slider('PTRATIO',10,30,20)
