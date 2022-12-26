import streamlit as st
import os
import math
import imblearn
import logging
import warnings
import statistics
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score , accuracy_score , precision_score, recall_score ,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from imblearn.combine import SMOTEENN
#%matplotlib inline




from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')














st.title("Electroencephalogram")

#st.sidebar.success("Select a page above")
#result=st.sidebar.button('Preprocess')


#train=st.sidebar.button('Train Model')
#test=st.sidebar.button('Test Model')


with st.form("analyze",clear_on_submit=False):
               x1=st.text_input("Input X1")
               x2=st.text_input("Input X2 ")
               x3=st.text_input("Input X3")
               x4=st.text_input("Input X4")
               x5=st.text_input("Input X5")
               submit=st.form_submit_button("Predict")

#data_sr=st.select_slider("Select Data Source", ["Load Data Automatically", "Upload Data from PC"])

data=pd.read_csv("Epileptic Seizure Recognition.csv")
#st.dataframe(load)

#if data_sr=="Load Data Automatically" :
    #dfload=pd.read_csv("Twitter.csv")
   # st.write('loading')
     

#if data_sr=="Upload Data from PC" :
 #    upload_file= st.file_uploader("Upload CSV file")
   
 
#if data_sr== "Upload Data from PC" and upload_file is not None:
         # Read the file to a dataframe using pandas
 #        data = pd.read_csv(upload_file)
  #       st.dataframe(data)
#upload_file= st.file_uploader("Upload CSV file") 

#if upload_file :
      
 #      data = pd.read_csv(upload_file) 
       
#     st.dataframe(data)       
#null_values = data.isnull().sum()
#null_values.to_numpy() #as we can see that there are no null values present on the dataset
data_1 = data.copy()
data_1.drop(['Unnamed','y'],axis=1,inplace=True)


data['y'].value_counts()



       #visualizing the only categorical column present in the dataset.
values = data['y'].value_counts()
#plt.figure(figsize=(7,7))
#values.plot(kind='pie',fontsize=17, autopct='%.2f')
#plt.legend(loc="best")
#plt.show()
       #it means all the categorical values in our dataset contains the equal amoung of balance.


       # plot these features in the same graph with stack plot
#fig, axs = plt.subplots(5, sharex=True, sharey=True)
#fig.set_size_inches(18, 24)
#labels = ["X15","X30","X45","X60","X75"]
#colors = ["r","g","b",'y',"k"]
#fig.suptitle('Visual representation of different channels when stacked independently', fontsize = 20)
       # loop over axes
#for i,ax in enumerate(axs):
 #         axs[i].plot(data.iloc[:,0],data[labels[i]],color=colors[i],label=labels[i])
  #        axs[i].legend(loc="upper right")

   #       plt.xlabel('total number of observation', fontsize = 20)
    #      plt.show()



#plt.figure(figsize=(10,10))
#this can help of provide us the general idea of how the waves are behaving 
#fig, axs = plt.subplots(1, sharex=True, sharey=True)
     #     plt.rcParams["figure.figsize"] = (20, 10)
      #    data.loc[:,::25].plot()
       #   plt.title("Visual representation different channels when stacked aganist each other")
        #  plt.xlabel("total number of values of x")
         # plt.ylabel("range of values of y")
          #plt.show()




       #   corr = data_1.corr()
       #   ax = sns.heatmap(
        #       corr, 
         #      vmin=-1, vmax=1, center=0,
          #       cmap=sns.diverging_palette(20, 220, n=200),
          #square=True
       #)




data_2 = data.drop(["Unnamed"],axis=1).copy()
data_2["Output"]= data_2.y == 0

data_2["Output"] = data_2["Output"].astype(int)

data_2.y.value_counts()

data_2['y'] = data_2['y'].replace([2,3,4,5],0)

data_2.y.value_counts() #we can see there is a mojor class imbalance problem in our dataset

          #plt.figure(figsize=(10,6),dpi=100)
          #sns.despine(left=True)
          #sns.scatterplot(x='X1', y='X2', hue = 'y', data=data_2)
          #plt.show()
#we can see the clear class imbalance problem present here
#data_2.head()
#data_2.y.value_counts()


          #X  = data_2.drop(['Output','y'], axis=1)
X=data_2.loc[:,['X1','X2','X3','X4','X5']]
y = data_2['y']

counter = Counter(y)
#finding out the 
print('Before',counter)
# oversampling the train dataset using SMOTE + ENN
smenn = SMOTEENN()
X_train1, y_train1 = smenn.fit_resample(X, y)

counter = Counter(y_train1)
print('After',counter)



#so we will start with dividing it into two parts/because with this method we cannot divide it into three parts
global X_train, X_test, y_train, y_test 
X_train, X_test, y_train, y_test = train_test_split(X_train1,y_train1,test_size=0.4,random_state=42)

#now we will be dividing it into further to get the validation set
global X_val,y_val
X_val, X_test, y_val, y_test = train_test_split(X_test,y_test,test_size=0.5,random_state=42)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#now we will going to scale the dataset
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


         


         

#intialising the nn
          #model = Sequential()

#layers
          #model.add(Dense(units=32,kernel_initializer='uniform',activation='relu',input_dim=5))

          #model.add(Dense(units=64,kernel_initializer='uniform',activation='relu'))
          #model.add(Dense(units=32,kernel_initializer='uniform',activation='relu'))
          #model.add(Dropout(0.25))
          #model.add(Dense(units=32,kernel_initializer='uniform',activation='relu'))
          #model.add(Dense(units=16,kernel_initializer='uniform',activation='relu'))
          #model.add(Dropout(0.5))
          #model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#finding out the summary of the model
          #model.summary()



#compiling the ann
          #model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#training the model
          #model_train = model.fit(X_train,y_train,batch_size=32,epochs=200,callbacks=[early_stopping],validation_split=0.2)

         #model.save('eeg_model.h5')
          
logreg = LogisticRegression()
model=logreg.fit(X_train, y_train)
          
from sklearn import metrics
import pickle
s = pickle.dumps(model)


         
          
          
if submit :
               
     one=np.array([[int(x1),int(x2),int(x3),int(x4),int(x5)],[-32,-39,-47,-37,-32]])
               
         
     predictions = logreg.predict(one)
                # print(np.round(predictions))
     print(predictions)         
     if predictions[0]==1 :       
         st.write('Prediction : Epileptic')
     if predictions[0]==0:
         st.write('Prediction : Non-Epileptic')
          
