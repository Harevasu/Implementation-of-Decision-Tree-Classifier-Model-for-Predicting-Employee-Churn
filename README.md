# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Algorithm

1. **Import necessary libraries.**  
2. **Load the dataset ("Employee.csv") into a pandas DataFrame.**  
3. **Inspect the dataset by viewing the first few rows and getting information about the data.**  
4. **Check for missing values and handle them (if any).**  
5. **Encode the categorical "salary" column using `LabelEncoder`.**  
6. **Select relevant features (`x`) and target variable (`y`) from the dataset.**  
7. **Split the dataset into training and testing sets using `train_test_split`.**  
8. **Initialize the `DecisionTreeClassifier` with the `entropy` criterion.**  
9. **Train the model using the training data.**  
10. **Make predictions on the test data.**  
11. **Evaluate the model using accuracy score.**  
12. **Print the accuracy and predict using a new data point.**
## Program:

### Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
### Developed by: HAREVASU S
### RegisterNumber: 212223230069
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![image](https://github.com/user-attachments/assets/b7af2a40-668d-4098-ba94-d85d6490675b)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
