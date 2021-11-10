import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from random import randint

dataset = np.loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]
new_label = np.float64(2)
for _ in range(150):
  Y[randint(0,700)] = new_label

seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier(objective="multi:softprob")

gamma = [0,1,2,3,4,5]
colsample_bytree = [x/100.0 for x in range(10, 102, 2)]
learning_rate = [x/1000.0 for x in range(1, 100, 5)]
max_depth = [1, 3, 5, 7, 9, 11]
n_estimators = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

param_grid = dict(gamma=gamma, colsample_bytree=colsample_bytree, learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))