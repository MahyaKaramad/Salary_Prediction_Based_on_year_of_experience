import pandas as pd
from matplotlib  import pyplot as plt
import numpy as np

######################################################pre_processing

#import data and show the first 5 rows
main_data = pd.read_csv("Salary_Data.csv")
print (main_data.head())


#check null values
print (main_data.isnull().sum())


#find out the relationship and find the model by showing on a plot 
#دلیل استفاده از 2 تا براکت چیست ؟
#این تضمین می‌کند که حتی اگر تنها یک ستون انتخاب شود، خروجی به شکل یک آرایه دو بعدی (یک جدول با یک ستون) باشد   
#در حالیکه اگه 1 عدد براکت بود مثل سری تک بعدی در نظر میگرفت

years = np.asanyarray(main_data[["YearsExperience"]])
salary =np.asanyarray(main_data[["Salary"]])

plt.scatter(salary, years , c="green" , marker=".")
plt.title  ('Raw Salary Data')
plt.xlabel ('Years')
plt.ylabel ('Salary')
plt.show   ()


# Split test and train data 
# random_state=20 means every time use the constant amount of nymbers and dont use random number  
from  sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(years, salary, test_size=0.2, random_state=20)


#################################################################################Processing


# Model selection 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)


#prediction .
a=""
while True:
    try:
        a = float(input("Please Enter your Years of Experience: "))    
        new_year = np.array([[a]])
        print("Predicted Salary = ", model.predict(new_year))
    except:
        print("Warnning: Plaese Enter a Float Number: ")


