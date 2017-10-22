import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import KFold

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Check status of missing data
n_missing = [None] * train.shape[1]
for i in range(train.shape[1]):
    n_missing[i] = len(np.where(train.iloc[:, i] == -1)[0])

plt.bar(range(train.shape[1]), n_missing)
plt.show()

# Define a function to change dtypes to save memory
def change_dtypes(data):
    for key in data.columns:
        if key.endswith('cat'):
            data[key] = data[key].astype('int16')
        elif key.endswith('bin'):
            data[key] = data[key].astype('uint8')
        elif key == 'id':
            data[key] = data[key].astype('int32')
        else:
            data[key] = data[key].astype('float32')
    return data

# Normalized Gini
def normal_gini(predict, d_train):
    labels = d_train.get_label()
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, predict)
    gini = 2 * auc - 1
    return ('gini', gini)
  
# Deal with missing values
#Method 1: Drop features that have more than 100,000 missing values
key_drop = []
for i in range(train.shape[1]):
    if n_missing[i] > 1E5:
        key_drop.append(train.columns[i])

for key in key_drop:
    train = train.drop(key, axis=1)
    test = test.drop(key, axis=1)
        
train = train.replace(-1, -999)
test = test.replace(-1, -999)
train = change_dtypes(train)
test = change_dtypes(test)

# OneHot
combine = pd.concat([train, test], axis=0)
cat_features = [feature_name for feature_name in combine.columns if feature_name.endswith('cat')]
combine = pd.get_dummies(data=combine, columns=cat_features)

train = combine[:train.shape[0]]
test = combine[train.shape[0]:]
test = test.drop('target', axis=1)

kf = KFold(n_splits=5, random_state=2017,shuffle=True)

X_train = train.drop(['id', 'target'], axis=1)
y_train = train.target

xgb_params = {'learning_rate': 0.02, 
              'max_depth': 4, 
              'subsample': 0.9, 
              'colsample_bytree': 0.9, 
              'min_child_weight': 2,
              'gamma': 0,
              'objective': 'binary:logistic', 
              'eval_metric': 'auc',
              'seed': 2017, 
              'silent': True}

d_test = xgb.DMatrix(test.drop('id', axis=1))
pred = []
for train_index, cv_index in kf.split(train):
    d_train = xgb.DMatrix(X_train.iloc[train_index, :], y_train.iloc[train_index])
    d_cv = xgb.DMatrix(X_train.iloc[cv_index, :], y_train.iloc[cv_index])
    watchlist = [(d_train, 'train'), (d_cv, 'cv')]

    model = xgb.train(xgb_params, d_train, 5000, evals=watchlist, feval=normal_gini, maximize=True, early_stopping_rounds=100, verbose_eval=50)
    pred.append(model.predict(d_test))
    
pred_avg = np.sum(np.asarray(pred), axis=0) / 5
df = pd.DataFrame({'id': test.id, 'target': pred_avg})
df.to_csv('result.csv', index=False)
