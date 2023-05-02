# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import sklearn
import pandas as pd
import matplotlib
#import pymongo
#import mongo

import numpy as np
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.figsize"] = (10,6)


import certifi
from pymongo import MongoClient
import pandas

MONGODB_URI = "mongodb://dataset_admin:QeQSfTHQAfW0XVZW@ac-rk0j8ox-shard-00-00.f5n4kgr.mongodb.net:27017,ac-rk0j8ox-shard-00-01.f5n4kgr.mongodb.net:27017,ac-rk0j8ox-shard-00-02.f5n4kgr.mongodb.net:27017/dataset?ssl=true&replicaSet=atlas-l45mpj-shard-0&authSource=admin&retryWrites=true&w=majority"
COLLECTION_NAME = "Bengaluru_House_Data"


def getDataFrame():
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    database = client.get_database()
    collection = database.get_collection(COLLECTION_NAME)
    cursor = collection.find()
    return pandas.DataFrame(cursor)

df1 = getDataFrame()
print(df1)
#df1 = pd.read_csv("Bengaluru_House_Data.csv")
#print(df1.head())


print(df1.shape)
# Shows Number of Rows and coloumns in the dataset

print(df1.groupby('area_type')['area_type'].agg('count'))
#Shows certain areas of the dataset

df2= df1.drop(['_id','area_type','society','balcony','availability'],axis='columns')
print(df1.head())

#We droped some of the columns which weren't necessary for the prediction

print(df2.head)
# This is how our dataset looks now


# Data Cleaning process starts here
print(df2.isnull().sum())
# isnull() function tell us the number of rows where column value is NULL

df3 = df2.dropna()
print(df3.isnull().sum)
print(df3.shape)
# Now we dropped the null values columns

print(df3['size'].unique())
#unique() function gives unique values of the column 'size'
# in the size column not all the values were alike
# some were BHK, some were Bedroom

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
# Now we create a new column BHK
# In which bhk shows only numbers instead of bedrooms

print(df3.head())

print(df3['bhk'].unique())
#this shows the values of bhk column only

print(df3[df3.bhk>20])
# this will only show the number of houses with more than 20 rooms

print(df3.total_sqft.unique())
# after using this function we found that there were many values in Range format rather than a single integer

def is_float(x):
    try:
        float(x)
    except :
        return False
    return True
# Than we find that in sqft columns all the values are FLOAT or Not


print(df3[~df3['total_sqft'].apply(is_float)])
# here i used a "~" negate operator which is like a is not thing
# it does the opposite of the is_float function


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return(float(tokens[0]) + float(tokens[1])/2)
    try:
        return float(x)
    except:
        return None

# In sqft some of the valuse were in a Range form 
# so this function convert returns the mean of the range 


df4 = df3.copy()
# It creates a copy of df3
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
# Here we applied the above function
print(df4)

print(df4.head(10))

df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
#This tell us the price per square feet
print(df5.head())

print(len(df5.location.unique()))
# this tell us the number of locations in our dataset

df5.location = df5.location.apply(lambda x: x.strip())
# This function strips the extra spaces/indentation from the location columns
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
# it give the statistics of the location
# sort function sorts the values in descending order
print(location_stats)



print(len(location_stats[location_stats<=10]))


location_stats_less_than_10 = location_stats[location_stats<=10]
print(location_stats_less_than_10)

print(len(df5.location.unique()))

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))

#this  puts the locations which have less than 10 data point as other locations
print(df5.head(10))


df5[df5.total_sqft/df5.bhk<300].head()

# this is threshold of 300 sqft per bedroom

df6 = df5[~(df5.total_sqft/df5.bhk<300)]
#Here we removed some of the outliers
df6.shape

df6.price_per_sqft.describe()

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m= np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
# created a function to filter mean and standard deviation
df7= remove_pps_outliers(df6)
# removed price per square feet outliers form df6
df7.shape
#print(df7.head(10))



def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (10,6)
    plt.scatter(bhk2.total_sqft, bhk2.price , color='blue', label = '2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price per Square Feet")
    plt.title(location)
    plt.legend()
    plt.show()

# This is a scatter plot chart which describe price of 2 and 3 bhk houses

plot_scatter_chart(df7, "Rajaji Nagar")


def remove_bhk_outlier(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis = 'index') 
# this function removes the value of 2bhk houses which are higher than the value of 3bhk houses

df8= remove_bhk_outlier (df7)
df8.shape

plot_scatter_chart(df8,"Hebbal")

import matplotlib
matplotlib.rcParams["figure.figsize"]=(10,6)
plt.hist(df8.price_per_sqft, rwidth=0.8)
plt.xlabel("Price Per Square Feet ")
plt.ylabel("Count")
plt.show()
# this is a histo graph 


df8.bath.unique()

df8[df8.bath>10]

plt.hist(df8.bath, rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("count")
plt.show()
# This shows a histo graph which shows the number count of bathrooms

df8[df8.bath>df8.bhk+2]
df9=df8[df8.bath<df8.bhk+2]
df9.shape

# this removes the property which have more bathrooms than the number of rooms

df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
df10.head(3)


dummies = pd.get_dummies(df10.location)
dummies.head(3)
# dummies convert the text to numeric column


df11 = pd.concat([df10,dummies.drop('other', axis = 'columns')],axis = 'columns')
df11.head(3)
# concatinated df10 and dummies and droped some columns

df12 = df11.drop('location', axis='columns')
df12.head(2)

X=df12.drop('price', axis = 'columns')
X.head()

y = df12.price
y.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

# we were traing the model 
# And get the score on the model 
# how good is our model

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
print(cross_val_score(LinearRegression(), X,y,cv=cv))
# Used K fold cross validation
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
    for algo_name, config in algos.items():
        print(algo_name, config)
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

print(find_best_model_using_gridsearchcv(X, y))
#  This function tells which algo is the best suitble
# and tells the best scores of all 3 algos



def predict_price(location,sqft,bath,bhk):
  loc_index = np.where(X.columns==location)[0][0]

  x = np.zeros(len(X.columns))
  x[0]=sqft
  x[1]=bath
  x[2]=bhk
  if loc_index >= 0:
    x[loc_index] = 1
  return lr_clf.predict([x])[0]



print(predict_price('1st Phase JP Nagar',1000,2,2))

print(" Chose Location from these locations only", X.columns)

location = input("Enter a valid Location")
sqft = int(input("Enter the area in Sqft"))
bath = int(input("Enter the number of Bathrooms"))
bhk = int(input("Enter number of Rooms"))

print("Price of the property is : ",predict_price(location,sqft,bath,bhk))


# now this functions finally predicts the price of the location 

"""

from tkinter import *
from functools import partial
def validatepassword(ep,cp):
    print("Entered password = ", ep.get())
    print("Confirmed password = 0", cp.get())
    if ep.get() == cp.get():
        Label(root, text = "password Confirmed").grid(row = 5, column = 0)
    else:
        Label(root, text = "Password Not matched").grid(row=5, column = 0)

root = Tk()
root.gepmetry('280*100')
root.title('Tkinter Password Validator')
Label1 = Label(root, text="Enter Password").grid(row = 0, columns = 0)
ep = StringVar()
passwordEntry = Entry(root, textvariable = ep, show = '*').grid(row = 0, columns = 1)

Label2 = Label(root, text = "Confirm Password").grid(row = 1, column=0)
cp = StringVar()
confirmpasswordEntry = Entry(root, textvariable =cp,show = '*' ).grid(row = 1, column = 1)

validatepassword = partial(Validatepassword, ep, cp)

checkbutton = Button(root, text = "check", command = validatepassword).grid(row = 4, column = 0)
root.mainloop()


#Tried doing Tkinter But there where many erros occuring so couldn't do it 
"""

