# Data-Analysis
## Heart Disease Data Analysis

### Loading relevant packages

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#### Load the data 

missing_values=["N/a","Not Applicable","na","",np.nan]
df=pd.read_csv(r"C:\Users\User\Desktop\Data Project\heart.csv",na_values=missing_values)
df.tail()

df.info()

#### The data contains no missing entries for all 14 attributes

#### The data types (Attribute Information)


df.dtypes

#### The data has 13 integer attributes and a single float variable

#### Renaming the columns for clarity

df = df.rename(columns = {'cp':'chest_pain_type', 'trestbps':'resting_blood_pressure', 'chol': 'cholesterol','fbs': 'fasting_blood_sugar', 
                       'restecg' : 'rest_electrocardiographic', 'thalach': 'max_heart_rate_achieved', 'exang': 'exercise_induced_angina',
                       'oldpeak': 'st_depression', 'slope': 'st_slope', 'ca':'num_major_vessels', 'thal': 'thalassemia'}, errors="raise")

#### The 5 number summary of the numerical variables

df.describe()

##### The age of patients seem to almost resemble no skewness since the mean and median are almost equal
#### The youngest patient is 29 years old and the oldest is 77 years
#### The lowest cholesterol level is 126 and the highest is 564. With the average person in the study having levels at about 246
#### The with the lowest resting blood pressure has about 94 and the highest reached 200. The average person has about 132
#### The maximum heart rate archived by an average patient in the study is about 150 and the minimum is 71 which is about half that of a an average patient. The maximum heart rate archieved is 202 , this implies the range is overly wide.

## Data Cleaning

df.isnull().sum()

#### The data has no missing entries

## Exploratory Data Analysis (EDA)

df.columns

df.describe()

##### The statistical 5 number summary of the model attributes

##### Number of transaction per sending partner
from matplotlib import rcParams
rcParams['figure.figsize'] = 8,6
pd.DataFrame(df["sex"].value_counts()).plot(kind="bar",figsize=(20,10),color='g', edgecolor='black', alpha=.8)
plt.style.use( 'fivethirtyeight')
plt.title("Number of transactions per sending partner")
plt.xlabel("Sending Partner(s)")
plt.ylabel("Count of transactions")
plt.show()

df.groupby("sex")["target"].count()

#### The are almost twice as many males as females. This might be due to the fact that men according to most literature are most
#### susciptible to heart diseases than their female counter parts

df["chest_pain_type"].unique()

plt.figure(figsize=(20,10))
chest_pain_type_count=df["chest_pain_type"].value_counts().tolist()
chest_pain_type_label=df["chest_pain_type"].value_counts().index
plt.pie(chest_pain_type_count,labels=chest_pain_type_label,
       autopct="%1.2f%%",wedgeprops={"edgecolor":"k"},
       textprops={"fontweight":"bold","size":10},shadow=True,
       explode=[0.1,0.1,0.1,0])
plt.show()

#### This seems to almost resemble a balanced classification problem (There are almost equal number of classes of those with heart diseases and the patients without.
#### There is less expected effort in terms of establishing class imbalances as these have an effect on model class predictions

plt.figure(figsize=(20,10))
target_count=df["target"].value_counts().tolist()
target_label=df["target"].value_counts().index
plt.pie(target_count,labels=target_label,
       autopct="%1.2f%%",wedgeprops={"edgecolor":"k"},
       textprops={"fontweight":"bold","size":10},shadow=True,
       explode=[0.1,0.1])
plt.show()

########Generate a dummy variable for flag
flag_dummies=pd.get_dummies(df["target"])
df1=pd.concat([df,flag_dummies],axis=1)
df["target"]

from matplotlib import rcParams
rcParams['figure.figsize'] = 20,10
sns.barplot(x="sex",y=["target"],data=df,color='green', edgecolor='black', alpha=.8)
xlim=plt.xlim()
plt.style.use('fivethirtyeight')
plt.title("The success ,failure and pending rates of the sending partners ")
plt.xticks(rotation=50)
plt.legend(["1","0"],loc="best")
plt.xlabel("The sending partners")
plt.ylabel("The sending success rate(s)")
plt.show()

df1.groupby("sex")["target"].value_counts()

df.head()



df["num_major_vessels"].unique()

sns.scatterplot(x="age",y="st_depression",data=df,color='green', edgecolor='black', alpha=.8)
xlim=plt.xlim()
plt.style.use('fivethirtyeight')
plt.title("The progression of st_depression by patient age")
plt.xlabel("The Patient Age")
plt.ylabel("ST Depression")
plt.show()

sns.scatterplot(x="age",y="max_heart_rate_achieved",data=df,color='green', edgecolor='black', alpha=.8)
xlim=plt.xlim()
plt.style.use('fivethirtyeight')
plt.title("The maximum achived heart rate by patient age")
plt.xlabel("The Patient Age")
plt.ylabel("The maximum heart rate achieved by patient")
plt.show()



#### The heatmap of the model attributes

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),cmap="RdBu",annot=True,vmin=0,vmax=1,linewidth=0.3)
title="Heatmap of the attributes correlation matrix"
plt.style.use('fivethirtyeight')
plt.title(title,loc="left")
plt.xlabel("Model Attributes")
plt.ylabel("Model Attributes")
plt.show()

# numerical fearures 6
num_feats = ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']

mypal= ['#FC05FB', '#FEAEFE', '#FCD2FC','#F3FEFA', '#B4FFE4','#3FFEBA']
L = len(num_feats)
ncol= 2
nrow= int(np.ceil(L/ncol))
#remove_last= (nrow * ncol) - L

fig, ax = plt.subplots(nrow, ncol, figsize=(16, 14),facecolor='#F6F5F4')   
fig.subplots_adjust(top=0.92)

i = 1
for col in num_feats:
    plt.subplot(nrow, ncol, i, facecolor='#F6F5F4')
    
    ax = sns.kdeplot(data=df, x=col, hue="target", multiple="stack", palette=mypal[1::4]) 
    ax.set_xlabel(col, fontsize=20)
    ax.set_ylabel("density", fontsize=20)
    sns.despine(right=True)
    sns.despine(offset=0, trim=False)
    
    i = i +1
plt.suptitle('Continous features kde-plot' ,fontsize = 24);

data = df.loc[df['target'] & df['cholesterol']]

plt.rcParams['figure.figsize'] = (15, 8)
ax = sns.boxplot(x = data['target'], y = data['cholesterol'], palette = 'inferno')
ax.set_xlabel(xlabel = 'Target', fontsize = 9)
ax.set_ylabel(ylabel = 'Age', fontsize = 9)
ax.set_title(label = 'Distribution of target by Age', fontsize = 20)
plt.xticks(rotation = 90)
plt.show()

