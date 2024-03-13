import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.datasets import fetch_openml
print('libraries imported!')

mnist = fetch_openml('mnist_784')
print('data load!')
X = mnist.data
y = mnist.target
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.15,
                                                shuffle=True,
                                                random_state=300)
print('split done!')

xg = xgb.XGBClassifier()
xg.fit(X_train,y_train.astype(np.int32))
print('training completed!')

pickle.dump(xg, open('model/pickle/image_model.pkl','wb'))
