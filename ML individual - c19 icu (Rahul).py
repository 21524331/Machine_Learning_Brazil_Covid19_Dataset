#!/usr/bin/env python
# coding: utf-8

# #                                    Exploratory Data Analysis 

# EDA is the basic step to analyze and perform intial examination.

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from natsort import index_natsorted


# Intially, we import the most fundamental libraries that will help us to perform EDA and Ml algorithm.

# In[3]:


data_icu = pd.read_csv('Kaggle_Sirio_Libanes_ICU_Prediction.csv')


# We use the pandas command to read our.csv file and declare our dataset as data_icu.

# In[4]:


data_icu.head()


# data_icu's first five rows are displayed by using .head() function.

# In[5]:


data_icu.tail()


# data_icu's last five rows are displayed by using .tail() function.

# In[6]:


(data_icu.shape)


# .shape was used to check the shape of data set like number of rows and columns,and it shows that (rows,coloumns): (1925,231) 

# In[7]:


def agr_perc_to_int(percentil):
    if percentil == "Above 90th":
        return (100)
    else:
        return(int("".join(c for c in str(percentil) if c.isdigit())))


# In[8]:


data_icu["AGE_PERCENTIL"] = data_icu.AGE_PERCENTIL.apply(lambda data_icu: agr_perc_to_int(data_icu))
set(data_icu["AGE_PERCENTIL"].values)


# In[9]:


def cat_window (window):
    if window == "ABOVE_12":
        return(13)
    else:
        return(int((window.split("-") [1])))
    
data_icu['WINDOW'] = data_icu['WINDOW'].apply(lambda x: cat_window(x))
data_icu['WINDOW'].isnull().sum()


# In the Above function it found that two columns were as objects (AGE_PERCENTIL and WINDOW), To avoid any numerical issues, it is suggested to convert them to numerical data or string to float. Above methods convert string into float.

# In[10]:


data_icu.info()


# Using the info() function, we can determine if the data is an integer or a float.

# In[11]:


data_icu.describe()


# data_icu describe() use for displays summary statistics for a python dataframe.

# In[12]:


data_icu.count()


# .count() function was used to check the values which are of given variable in our data set(data_icu).

# In[13]:


data_icu.nunique()


# The nunique()function returns the number of unique values for each column. 

# In[14]:


data_icu.columns


# We observed columns in datasets by using the.columns function.

# # Data Dublication

# # Data Quality Issues 

# In[15]:


data_icu.drop_duplicates()
data_icu.shape


# In[16]:


data_icu.duplicated(subset=None, keep="first")


# .drop_duplicates method is used to remove duplicate or double column and its show that in our data set their is no duplicate column.

# In[17]:


data_icu = data_icu.drop_duplicates(keep='first')


# In[18]:


# Use for dropping Null values
data = data_icu.dropna(axis=1,how="all")
data_icu.shape


# .dropna method are used to drop  Null values and this function remove a entire column in which every value is Null.So, its  shown that in our data set thier no such column which contain  every value null.

# In[19]:


data_icu.isnull().sum()


# .isnull() function is used to check null values of our data set and using it with .sum() the null values will repsent in tabulor form as shown above . It appear that four variable countain null values . 
# 

# ## Missing Values

# In[20]:


def _impute_missing_data(data_icu):
    return data_icu.replace(-1, np.nan)
data_icu = _impute_missing_data(data_icu)


# In[21]:


print('NaN values = ', data_icu.isnull().sum().sum())
print("""""")
      
vars_with_missing = []
      
for feature in data_icu.columns:
      missings = data_icu[feature].isna().sum()
      
      if missings > 0 :
          vars_with_missing.append(feature)
          missings_perc = missings / data_icu.shape[0]
          
          print ('Variable {} has{} records ({:.2%}) with missing values.'.format(feature, missings, missings_perc))
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))


# Above method is used to checked the number of NaN values or missing values.

# # Missing Values

# In[22]:


pd.DataFrame({"Columns": data_icu.columns,"Missing_values":((data.isna()).sum()/data_icu.shape[0])*100})


# Above table represent how many missing values there are in our dataset in percent.

# # Barplot 

# In[23]:


import missingno as msno
msno.bar(data_icu)


# We import missingno library for visualization our data. 
# we plot bar graph for visualization of missing values.

# # Heatmap

# In[24]:


msno.heatmap(data_icu)


#  Heat map, which is a two-dimensional visual representation of data, each value in a matrix is represented by a different hue.

# In[25]:


corelation = data_icu.corr()
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)


# Because of large amount of data, many of Nan Values, Unable to find out Correlation between any columns.

# # Pivot

# In[26]:


pd.pivot_table(data_icu, index=['ICU', 'GENDER'], columns = ['AGE_ABOVE65'], aggfunc=len)


#  - In ICU
# 
# 0 - Patient not admitted in ICU  
# 1 - Patient admitted in ICU
# 
# 0 - Represent Patient is not Critical.  
# 1 - Represent Patient is in Critical condition due to Covid-19.
# 
# - In GENDER   
# 0 - Male   
# 1 - Female
# 
# Outside of ICU, Females are less critical than Males.  
# More females than males were admitted to the ICU.

# # UNIVARIANT EXPLORATION

# The word "Uni" means "one," therefore "univariate analysis" refers to the study of only one variable at a time.

# For univariant observation we use distplot,data distribution of a variable against the density distribution.

# In[27]:


sns.distplot(data_icu['TEMPERATURE_DIFF'])
plt.show()


# Temperature density of the patients countinously surge and reached to 1.25 and then follow decline pattern in betwwen -0.5 to 0.C

# In[28]:


sns.distplot(data_icu['OXYGEN_SATURATION_DIFF'])
plt.show()


# Patient Oxygen saturation Density lies in between -0.1 to 0.5 and reached to peak of around 2.75.

# In[29]:


sns.distplot(data_icu['BLOODPRESSURE_DIASTOLIC_DIFF_REL'])
plt.show()


# The highest density of a Bloodpressure_diastolic_diff_rel reached at a peak of 1.3 .

# In[30]:


sns.distplot(data_icu['BLOODPRESSURE_SISTOLIC_DIFF_REL'])
plt.show()


# Here the Density peak is 1.2.

# In[31]:


sns.distplot(data_icu['HEART_RATE_DIFF_REL'])
plt.show()


# The maximum density for HEART_RATE_DIFF_REL IS 1.5.

# In[32]:


sns.distplot(data_icu['RESPIRATORY_RATE_DIFF_REL'])
plt.show()


# The maximun density for RESPIRATORY_RATE_DIFF_REL is 1.2.

# In[33]:


sns.distplot(data_icu['TEMPERATURE_DIFF_REL'])
plt.show()


# The maximun density for TEMPERATURE_DIFF_REL is 1.25.

# In[34]:


sns.distplot(data_icu['OXYGEN_SATURATION_DIFF_REL'])
plt.show()


# The maximun density for OXYGEN_SATURATION_DIFF_REL is 2.8.

# In[35]:


sns.distplot(data['ICU'])
plt.show()


# The maximun density for ICU is 3.

# In[36]:


sns.distplot(data_icu['PATIENT_VISIT_IDENTIFIER'])
plt.show()


# In[37]:


a=data_icu['ICU'].value_counts()
plt.pie(a,labels = ['NON-ICU', 'ICU'])
plt.show()


# According to pie chart more then half patients are not required ICU beds .

# # BIVARIENT EXPLORATION 

# Analyzing two variables simultaneously is known as bivariate analysis.

# In[38]:


sns.boxplot(x="GENDER" , y="AGE_ABOVE65",data = data_icu)


# Men and women over the age of 65 are both equally affected. covid -19.

# In[39]:


sns.boxplot(x="ICU", y="AGE_ABOVE65" , data = data_icu)


# ICU and non-ICU patients with affected COVID rates were equal.

# In[40]:


age = sns.countplot(x='AGE_ABOVE65' , hue='GENDER' , data=data_icu)
for p in age.patches:
    height = p.get_height()
    age.text(p.get_x() + p.get_width()/2. , height + 0.1, height, ha="center")


# In[41]:


icu=sns.countplot(x="ICU", hue = "GENDER", data = data_icu)
for p in icu.patches:
    height = p.get_height()
    icu.text(p.get_x() + p.get_width()/2., height + 0.1,height, ha="center")


# In compare to Female patient, Male covid patient are highly admitted.

# # Data Preparation

# DATA CLEANING

# It is necessary to clean our data  before performing ml algorithm.

# In[42]:


drop_cols = ['TEMPRETURE_DIFF' , 'OXYGEN_SATURATION_DIFF', 'BLODDPRESSURE_DIASTOLIC_DIFF_REL' 'BLOODPRESSURE_SISTOLIC_DIFF_REL']


# In[43]:


data_icu


# In[44]:


dataset = data_icu.copy()


# In[45]:


data = data_icu.fillna(0)


# In[46]:


dataset = data.fillna(0)


# The fillna() function substitutes a given value for any NULL values. We replaced Null values with 0. 

# In[47]:


import missingno as msno
msno.bar(data)


# We import missingno library for visualization our data. we plot bar graph for visualization of missing values.After filling nun value. 

# # Identified Outliers

# An outlier is a data point in a data set that is distant from all other observations. A data point that lies outside the overall distribution of the dataset.

# In[48]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=2, cols=4)
fig.add_trace(go.Box(y=data_icu['BLOODPRESSURE_SISTOLIC_MAX'],name='BLOODPRESSURE_SISTOLIC_MAX'),row=1,col=1)
fig.add_trace(go.Box(y=data_icu['HEART_RATE_MAX'],name='HEART_RATE_MAX'),row=1,col=2)
fig.add_trace(go.Box(y=data_icu['RESPIRATORY_RATE_MAX'],name='RESPIRATORY_RATE_MAX'),row=1,col=3)
fig.add_trace(go.Box(y=data_icu['TEMPERATURE_MAX'],name='TEMPERATURE_MAX'),row=1,col=4)
fig.add_trace(go.Box(y=data_icu['OXYGEN_SATURATION_MAX'],name='OXYGEN_SATURATION_MAX'),row=2,col=1)
fig.add_trace(go.Box(y=data_icu['BLOODPRESSURE_DIASTOLIC_DIFF'],name='BLOODPRESSURE_DIASTOLIC_DIFF'),row=2,col=2)
fig.add_trace(go.Box(y=data_icu['BLOODPRESSURE_SISTOLIC_DIFF'],name='BLODDPRESSURE_SISTOLIC_DIFF'),row=2,col=3)
fig.add_trace(go.Box(y=data_icu['HEART_RATE_DIFF'],name='HEART_RATE_DIFF'),row=2,col=4)
fig.show()


# In[49]:


data_icu.BLOODPRESSURE_SISTOLIC_MAX.mean()
data_icu.BLOODPRESSURE_SISTOLIC_MAX.std()
data_icu.BLOODPRESSURE_SISTOLIC_MAX.describe()


# In[50]:


upper_limit = data.BLOODPRESSURE_SISTOLIC_MAX.mean() + 3*data_icu.BLOODPRESSURE_SISTOLIC_MAX.std()
upper_limit


# According to our result upper result is 0.60

# In[51]:


lower_limit = data_icu.BLOODPRESSURE_SISTOLIC_MAX.mean() - 3*data_icu.BLOODPRESSURE_SISTOLIC_MAX.std()
lower_limit


# According to our result, lower limit -1.25 

# # ML ALgorithm.

# In[52]:


x=dataset.drop('ICU', axis=1)
x


# In[53]:


y=dataset['ICU']
y


# # Split the data into Two part

# we are spliting our data for running our machine learning modelieng.

# In[54]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test=train_test_split(x,y,test_size=0.2, random_state=95)


# In[55]:


x_train


# In[56]:


y_train


# In[57]:


x_test


# In[58]:


y_test


# #  Decision Tree

# One of the strongest and most well-liked algorithms is the decision tree. The decision-tree algorithm is a type of supervised learning method. It functions with output variables that are categorised and continuous.

# In[59]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
dtc=DecisionTreeClassifier()
dtc.fit(x_train, y_train)


# In[60]:


y_pred1=dtc.predict(x_test)


# In[61]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_pred1,y_test)


# The accuracy of this model is 0.78

# In[62]:


confusion_matrix(y_pred1,y_test)


# In[63]:


plt.figure(figsize=(40,40)) # set plot size (denoted in inches)
plot_tree(dtc, filled=True, fontsize=10)
plt.show()


# # Random Forest 

# Random forest selects a random sample from the training set, creates a decision tree for it and gets a prediction; it repeats this operation for the assigned number of the trees, performs a vote for each prediction, and takes the result with the majority of votes (in case of classification) or the average

# In[64]:


from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier()
regressor.fit(x_train, y_train)


# In[65]:


y_pred=regressor.predict(x_test)


# In[66]:


from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:' , metrics.mean_squared_error(y_pred,y_test))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('R-Squared', r2_score(y_pred, y_test))


# In[67]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_pred,y_test)


# The accuracy for this model is 0.841

# In[68]:


confusion_matrix (y_pred,y_test)


# In[69]:


plt.figure(figsize=(8,6))
plt.plot(y_test,y_test,color='deeppink')
plt.scatter(y_test,y_pred,color='dodgerblue')
plt.xlabel('Actual Target Value',fontsize=15)
plt.ylabel('Predicted Traget Value',fontsize=15)
plt.title('Random Forest Regressor',fontsize=14)
plt.show()


# # XG Boost

# XGBoost is used for supervised learning problems, where we use the training data (with multiple features) to predict a target variable.

# In[70]:


from xgboost import XGBClassifier
Classifier = XGBClassifier()


# In[71]:


Classifier.fit(x_train, y_train)


# In[72]:


y_pred2=Classifier.predict(x_test)


# In[73]:


accuracy_score(y_pred2,y_test)


# The accuracy for this model is 0.88

# In[74]:


confusion_matrix(y_pred2,y_test)


# # SVM 

# Support Vector Machine” (SVM) is a supervised machine learning algorithm that can be used for both classification or regression challenges. However, it is mostly used in classification problems.

# In[75]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[76]:


svm = SVC()

param_grid = { 
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001]
}

cv_svm = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
cv_svm.fit(x_train, y_train.values.ravel())

print("Support Vector Machines Model best params: ", cv_svm.best_params_)

# Training model with best params
best_params = cv_svm.best_params_

svm_best = SVC(random_state = 42, 
              C = best_params['C'], 
              gamma = best_params['gamma'])

svm_best.fit(x_train, y_train)
y_pred = svm_best.predict(x_test)

# Evaluating the model
print("Accuracy for Support Vector Machines is : ", round(accuracy_score(y_test, y_pred), 2))

print("\n\nClassification report for Support Vector Machines:")
print(classification_report(y_test, y_pred))

print("\n\nConfusion matrix for Support Vector Machines:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)


# The accuracy for this model is 0.83.

# # Modeling and Model Evaluation

# # Ensemble Voting Method 

# In[77]:


clfs = {"SVM":SVC(kernel='rbf', probability=True),
       "DecisionTree":DecisionTreeClassifier(),
       "RandomForest":RandomForestClassifier(),
       "XGBoost":XGBClassifier(verbosity=0)}


# In[78]:


import math
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
def model_fit(clfs):
    
    fitted_model = {}
    model_result = pd.DataFrame()
    for model_name, model in clfs.items():
        model.fit(x_train,y_train)
        fitted_model.update({model_name:model})
        y_pred =model.predict(x_test)
        model_dict = {}
        model_dict['1.Algorithm'] = model_name
        model_dict['2.Accuracy'] = round(accuracy_score(y_test, y_pred),3)
        model_dict['3.Precision'] = round(precision_score(y_test,y_pred),3)
        model_dict['4.Recall'] = round(recall_score(y_test,y_pred),3)
        model_dict['5.F1'] = round(f1_score(y_test, y_pred),3)
        model_dict['6.ROC'] = round(roc_auc_score(y_test, y_pred),3)
        model_result = model_result.append(model_dict,ignore_index=True)
    return fitted_model, model_result
     


# In[79]:


fitted_model, model_result = model_fit(clfs)
model_result.sort_values(by=['2.Accuracy'],ascending=False)


# In[80]:


model_ordered = []
weights = []
i=1
for model_name in model_result['1.Algorithm'][
    index_natsorted(model_result['2.Accuracy'],reverse=False)]:
    model_ordered.append([model_name,clfs.get(model_name)])
    weights.append(math.exp(i))
    i+=0.8


# In[81]:


plt.plot(weights)
plt.show()


# In[82]:


weights


# # Conclusion 

# The best algorithm for brazil covid - 19 dataset is XGBoost, providing  the best Performance.

# # Executive Summary

# #Student ID: 21524331, Rahul Ladhani 
# 
# 
# Introduction 
# In this model, We will utilize the Hospital Sirio-Libanes data set to determine if the covid patient needs an ICU bed or not. A machine learning strategy was used to maximize the usage of ICU beds because there were not enough of them due to the covid pandemic crisis in Brazil. Therefore, depending on the patient's present medical status, we will utilize the data set to attempt to categorise whether the patient would need an ICU bed.
# 
# Basically, the ML model divided into two parts
# 1 EDA – Data quality issues, univariant, bivariant, data preparation(data cleaning ).  
# 2 ML algorithm - Decision Tree, Random forest, XG Boost, and SVM. Modeling and Model Evaluation
# 
# Exploratory Data Analysis (EDA)
# In order to recognize the inconsistencies and missing values in the data set, we performed exploratory data analysis on it. The following is a list of some of the data set's examination:
# The data set contained 1925 raw, 231 columns. Having some missing values. We found some objects were strings(AGE_Percitile and ICU), so we converted them into float. We plot some graphs for better visualization. Bar plot, heat map and Pivot.
# Univariant exploration - The word "Uni" means "one," therefore "univariate analysis" refers to the study of only one variable at a time. For univariant observation, we use distplot, data distribution of a variable against the density distribution. We visualized variables to understand the data distribution.
# Bivariant exploration - Analysing two variables simultaneously is known as bivariate analysis. We plot graphs between, gender and age above 65, ICU and age. For an understanding of the distribution of ICU beds.
# 
# Data preparation:  we clean the data for our machine-learning modelling by using the fillina method which replaces nun values with 0. Then plot a graph to understand the nun values after fillina method.
# 
# Identified Outliers:
# An outlier is a data point in a data set that is distant from all other observations. A data point that lies outside the overall distribution of the dataset. We plot Outliers for variables and check upper and lower values for BLOODPRESSURE_SISTOLIC_MAX. 
# 
# 
# Machine learning (ML):
# To perform the ml algorithm first we have to train the data model into x and y – test and train, In order to train the model, 80% of the data set was used, while the remaining 20% was used for model assessment(test).
# 
# The sklearn machine learning library has been used to generate machine learning classification models. The models we will design are:
# 
# 1.	 Decision Tree Classifier
# 2.	 Random Forest
# 3.	 XG Boost
# 4.	 Support Vector Machines
# 
# 
# 1 . Decision Tree Classifier :The Decision Tree model is performing is a bit less accurate. The accuracy we get is 
#  0.78. we plotted the graph and show the confusion matrix as well.
# 
# 2. Random Forest: The Random Forest model performs with different accuracy for both classes like the Decision tree.  But the accuracy is high 0.85.
# 
# 3 . XG boost : XG boost model shows the highest accuracy compared to all models and the accuracy is 0.88.
#     
# 4 Support Vector Machines: The SVM model is predicted a little bit accurately for the patients who don’t require ICU beds. The overall accuracy is 0.83. 
# 
# Modeling and Model Evaluation
# Ensemble Voting Method
# 
# We performed an ensemble voting method for representing our models and their feature in a table and graph.
# 
# 
# Conclusion :
# In the project, We were able to use machine learning technology to determine if the patient would need an ICU bed upon hospital admission. We were able to achieve the best accuracy of 88% in predicting if the patient admitted to the hospital would eventually be required to admit to the ICU. In order to improve the accuracy and predictions from the model, we may strive to collect additional data and aim to attain even greater accuracy in future work. 
# 
# 
# 

# 
