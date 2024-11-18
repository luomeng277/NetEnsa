import matplotlib.pyplot as plt
import xgboost
import shap
import pandas as pd
import shap
import matplotlib.pyplot as plt
from numpy import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

path ='../result/'
X1 = pd.read_csv(path + 'aden.csv', index_col=0).T
X1_col = pd.read_csv(path + 'aden.csv', index_col=0).T.columns
label_data = pd.read_csv(path + 'label.csv', index_col=0)


# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X1, label_data, test_size=0.3, random_state=50)

# 定义交叉验证对象
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=50)

# 初始化指标列表
acc_x = []
f1_x = []
auc_x = []
acc_r = []
f1_r = []
auc_r = []
acc_s = []
f1_s = []
auc_s = []
num_runs = 20
for run in range(num_runs):
    # 重置指标累加变量
    acc_x_run = []
    f1_x_run = []
    auc_x_run = []
    acc_r_run = []
    f1_r_run = []
    auc_r_run = []
    acc_s_run = []
    f1_s_run = []
    auc_s_run = []
# # 在循环中进行交叉验证的训练和测试
for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
    train_x1, train_y = X_train.iloc[train_index], y_train.iloc[train_index]
    test_x1, test_y = X_train.iloc[test_index], y_train.iloc[test_index]

    # 使用XGBClassifier进行训练和预测
    gnb1x = XGBClassifier()
    gnb1x.fit(train_x1, train_y)
    y_pred1x = gnb1x.predict(test_x1)
    y_pro1x = gnb1x.predict_proba(test_x1)[:, 1]
    acc_x.append(accuracy_score(test_y, y_pred1x))
    f1_x.append(f1_score(test_y, y_pred1x))
    auc_x.append(roc_auc_score(test_y, y_pro1x))

#     # 使用RandomForestClassifier进行训练和预测
    gnb1r = RandomForestClassifier(random_state=0)
    gnb1r.fit(train_x1, train_y)
    y_pred1r = gnb1r.predict(test_x1)
    y_pro1r = gnb1r.predict_proba(test_x1)[:, 1]
    acc_r.append(accuracy_score(test_y, y_pred1r))
    f1_r.append(f1_score(test_y, y_pred1r))
    auc_r.append(roc_auc_score(test_y, y_pro1r))

    # 使用SVC进行训练和预测
    gnb1s = SVC(probability=True, kernel='rbf')
    gnb1s.fit(train_x1, train_y)
    y_pred1s = gnb1s.predict(test_x1)
    y_pro1s = gnb1s.predict_proba(test_x1)[:, 1]
    acc_s.append(accuracy_score(test_y, y_pred1s))
    f1_s.append(f1_score(test_y, y_pred1s))
    auc_s.append(roc_auc_score(test_y, y_pro1s))
#
print('xgboost :')
print('acc:', sum(acc_x) / len(acc_x))
print('f1:', sum(f1_x) / len(f1_x))
print('auc:', sum(auc_x) / len(auc_x))
print('RandomForest :')
print('acc:', sum(acc_r) / len(acc_r))
print('f1:', sum(f1_r) / len(f1_r))
print('auc:', sum(auc_r) / len(auc_r))



print('SVM :')
print('acc:', sum(acc_s) / len(acc_s))
print('f1:', sum(f1_s) / len(f1_s))
print('auc:', sum(auc_s) / len(auc_s))




