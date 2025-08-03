import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


x, y = load_breast_cancer(return_X_y=True, as_frame=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

feorest = RandomForestClassifier(n_estimators=400, max_leaf_nodes=16, n_jobs=-1)
feorest.fit(x_train, y_train)

y_forest_pred = feorest.predict(x_test)

accuracy = accuracy_score(y_forest_pred, y_test)
print(accuracy)