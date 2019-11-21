# --------------
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data

# print the first 5 rows of the dataset

# Split the data into independent and target variable

# Split the data into train and test data sets



# --------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Assign the Randomforrest classifier to a variable rf
rf = RandomForestClassifier()
# Train the model
rf.fit(X_train,y_train)
# Predict the class on the test data
y_pred = rf.predict(X_test)


# --------------
from sklearn.metrics import accuracy_score,mean_absolute_error

# Accuracy score
accuracy_score(y_test,y_pred)


