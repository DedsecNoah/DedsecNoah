import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


data = pd.read_csv('data.csv')

data = data[["Total_Applicants", "Total_Enrolled", "Year"]]
predict = "Total_Enrolled"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print("Accuracy: ", acc)


predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print("Predicted Enrolled: ", predictions[i], x_test[i], "\nOriginally Enrolled: ", y_test[i])

