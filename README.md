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


#Practise
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

missing_values=["N/a","Not Applicable","na","",np.nan]

df=pd.read_excel(r"C:\Users\User\Desktop\Machine Learning\MF AFRICA\Book1.xlsx",na_values=missing_values)

df

df.columns

df.info()

df.columns


df=df[[ 'id','operation','sending_partner', 'receiving_partner', 
        
        'send_amount',
 'flag', 'receive_amount',
         'transaction_datetime']]

df.head()

df.isna().sum()

df[df["flag"].isnull()].tail()

df.isnull().sum()

df[df["receive_amount"].isnull()]

df.head()

df.tail()

df.dropna(axis=0,inplace=True)

df["flag"].fillna("Not Applicable",inplace=True)
df["status_of_receiving_account"].fillna("Not Applicable",inplace=True)

df.isnull().sum()

df

df.isnull().sum()

df.describe()

df[df["flag"].isnull()]

df[df["flag"].isna()]

df["flag"].value_counts()

df

df.isnull().sum()

df.head()

dup_df=df.duplicated().any()
print("Are there any mssing values?",dup_df)

df=df.drop_duplicates()

df.duplicated().any()

df.isnull().sum()

df.dropna(axis=0)

df.isnull().any()

df=df[df["flag"].isnull()]

df.dropna()

df.tail()

df["flag"].fillna("Not Applicable",inplace=True)
df["receiving_partner"].fillna("Not Applicable",inplace=True)
df["send_amount"].fillna("Not Applicable",inplace=True)
df["receive_amount"].fillna("Not Applicable",inplace=True)
df["transaction_datetime"].fillna("Not Applicable",inplace=True)
df["sending_partner"].fillna("Not Applicable",inplace=True)
df["status_of_receiving_account"].fillna("Not Applicable",inplace=True)

df.isna().sum()

df["transaction_datetime"].tail(10)

df.describe(include="all")

df["status_of_receiving_account"].value_counts()	

df["flag"].value_counts()

pd.DataFrame(df["flag"].value_counts()).plot(kind="bar",figsize=(20,10))

pd.DataFrame(df["receiving_partner"].value_counts()).plot(kind="bar",figsize=(20,10))

df.dtypes



pd.DataFrame(df["sending_partner"].value_counts()).plot(kind="bar",figsize=(20,10))

df=df[df["flag"]=="Success"]
df.groupby("receive_amount").mean()

df.dtypes

df.nlargest(10,"receive_amount")[["receiving_partner","receive_amount"]]\
.set_index("receiving_partner")

pd.DataFrame(df["receive_amount"].sum()).plot(kind="bar",figsize=(20,10))
plt.xlabel("Receiving Partners")
plt.legend("receiving_partner")
plt.show()

sns.heatmap(df.corr())

df.groupby("sending_partner")["send_amount"].mean().sort_values(ascending=False).head(10)

kj=df.groupby("receiving_partner")["receive_amount"].mean().sort_values(ascending=False)

kj

df.describe()

df[df["receive_amount"]>=1000000]["receiving_partner"]

df.describe(
)

plt.figure(figsize=(12,6))
cond=df["send_amount"].sum()>=2.600233e+04
sns.barplot(y=df["receiving_partner"],x=df["receive_amount"])
xlim=plt.xlim()
plt.xlim(xlim)
plt.xlabel("Received amount (in million of Rands)")
plt.ylabel("")
plt.yticks(rotation=0.5)
plt.title("Receiving Partner by received amount")
plt.show()

loop=df.groupby("receiving_partner")["receive_amount"].mean()

loop.head()

df.groupby("receiving_partner")["flag"].value_counts().sort_values(ascending=False).head(10
)

df.groupby("sending_partner")["flag"].value_counts().sort_values(ascending=False).tail()

df.groupby("sending_partner")["flag"].value_counts().sort_values(ascending=False).head()

pd.DataFrame(df["receiving_partner"].value_counts()).plot(kind="bar",figsize=(20,10))

df.columns

df.groupby("operation")["send_amount"].sum()

plt.figure(figsize=(12,6))
sns.barplot(y=df["operation"].sort_values(ascending=False),x=df["send_amount"])
plt.title("Operation by sent amount")
plt.show()

df.groupby("operation")["receive_amount"].sum()

plt.figure(figsize=(12,6))
sns.barplot(y=df["operation"],x=df["receive_amount"].sort_values(ascending=False))
plt.xlabel("Received amounts in (Rands)")
plt.ylabel("Operations")
plt.yticks(rotation=60
           
          )
plt.title("Operation by received amount")
plt.show()

df.groupby("operation")["flag"].value_counts()

df.columns

df.groupby('transaction_datetime')['sending_partner'].value_counts()

bar=plt.bar(df["operation"],df["operation"].value_counts())
bar[0].set_hatch("/")
bar[1].set_hatch("*")
bar[2].set_hatch("-")
bar[3].set_hatch("|")
bar[4].set_hatch("")
plt.xlabel("MFS Operations")
plt.xticks(rotation=0)
plt.ylabel("Count of transactions per operation")
plt.title("Number of transactions by operation")
plt.legend()
plt.show()

data=df.groupby("sending_partner")["send_amount"==>2.524393e+04]

data.head()

cond=df["send_amount"].mean()>2.524393e+04
plt.bar(df["send_amount"].mean()[cond==False],df["sending_partner"][cond==False])
plt.ylim(ylim)
plt.xlim(xlim)
plt.show()


df.columns

plt.figure(figsize=(12,6))
sns.barplot(y=df["receiving_partner"],x=df["receive_amount"])
xlim=plt.xlim()
plt.xlim(xlim)
plt.xlabel("Received amount (in million of Rands)")
plt.ylabel("")
plt.yticks(rotation=0.5)
plt.title("Receiving Partner by received amount")
plt.show()

place_df=pd.DataFrame(df["sending_partner"],df["send_amount"])

df.describe()

place_df

df[df["send_amount"].mean()>=2.524393e+04]["sending_partner"]

big_transact=df[df["send_amount"]>2.524393e+04]

big_transact.

data=df[df["flag"]=="Success"]

data

data.groupby("sending_partner")["send_amount"].describe()

data.groupby("receiving_partner")["receive_amount"].describe()

df.describe()

non_zero=df[df["send_amount"]>2.524393e+04]

non_zero


plt.figure(figsize=(12,6))
sns.barplot(x="send_amount",y="sending_partner",data=non_zero)
xlim=plt.xlim()
plt.title("Plot of the sums of send partner amounts that were above average")
plt.xlabel("Partner send amounts (in Thousand of Rands)")
plt.ylabel("Send partners")
plt.show()

nn_zero=df[df["receive_amount"]>4.408000e+04]

nn_zero

plt.figure(figsize=(20,10))
sns.barplot(x="receive_amount",y="receiving_partner",data=nn_zero)
xlim=plt.xlim()
plt.title("Plot of the sums of receive partner amounts that were above average")
plt.xlabel("Partner receive amounts (in Thousand of Rands)")
plt.ylabel("Receive partners")
plt.show()

suc_data=df[df["flag"]=="Success"]

suc_data.groupby("sending_partner")["send_amount"].describe()

suc_data.describe()

nw=suc_data[suc_data["send_amount"]>]



plt.figure(figsize=(20,10))
sns.barplot(x="send_amount",y="sending_partner",data=suc_data)
xlim=plt.xlim()
plt.title("Plot of the sums of successful send partner amounts")
plt.xlabel("Partner send amounts (in Thousand of Rands)")
plt.ylabel("Send partners")
plt.show()

fail_data=df[df["flag"]=="Fail"]

fail_data

plt.figure(figsize=(20,10))
sns.barplot(x="send_amount",y="sending_partner",data=fail_data)
xlim=plt.xlim()
plt.title("Plot of the sums of fail send partner amounts that were above average")
plt.xlabel("Partner send amounts (in Thousand of Rands)")
plt.ylabel("Send partners")
plt.show()

data2=suc_data.groupby("sending_partner")["send_amount"].mean()

data2

data3=data2[data2>6.699600e+04]

data3

data3.plot(kind="bar",figsize=(20,10))

plt.figure(figsize=(20,10))
data3.plot(kind="bar",figsize=(20,10))
xlim=plt.xlim()
plt.title("Plot of the sums of fail send partner amounts that were above average")
plt.xlabel("Partner send amounts (in Thousand of Rands)")
plt.xticks(rotation=20)
plt.ylabel("Send partners")
plt.show()


