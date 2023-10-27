import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
data_df=pd.read_csv("/content/creditfaurd.csv")
data_df.head()
data_df.shape #return no rows and cols
data_df[['Amount','Time','Class']].describe()
data_df.isna().any()   #if there any null values its return ture otherwise false
data_df['Class'].tail(10) #the last 10 records of your dataset
nf_count=0
notFaurd=data_df['Class']
for i in range(len(notFaurd)):
  if notFaurd[i]==0:
    nf_count=nf_count+1
nf_count
per_nf=(nf_count/len(notFaurd))*100
print("percentage of total not faurd in the dataset:",per_nf)
from numpy.ma import flatten_structured_array
f_count=0
Faurd=data_df['Class']
for i in range(len(Faurd)):
  if Faurd[i]==1:
    f_count=f_count+1
f_count
per_f=(f_count/len(Faurd))*100
print("percentage of total faurd in the dataset",per_f)
plt.title("count plot for faurd vs genuine transaction")
sns.countplot(x= 'Class',data = data_df,palette='Blues',edgecolor='w')
x=data_df['Amount']
y=data_df['Time']
plt.plot(x,y)
plt.title('Time vs amount')
fig,ax=plt.subplots(figsize=(16,8))
ax.scatter(data_df['Amount'],data_df['Time'])
ax.set_xlabel('Amount')
ax.set_ylabel('Time')
plt.show()
correlation_metrics=data_df.corr()
fig=plt.figure(figsize=(14,9))
sns.heatmap(correlation_metrics,vmax=.9,square=True)
plt.show()
x=data_df.drop(['Class'],axis=1) # drop the target variable
y=data_df['Class']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
logisticreg=LogisticRegression()
logisticreg.fit(xtrain,ytrain)
y_pred=logisticreg.predict(xtest)
accuracy=logisticreg.score(xtest,ytest)
accuracy
cm=metrics.confusion_matrix(ytest,y_pred)
print(cm)
print("Accuracy score of the logistic regression:",accuracy*100,'%')