
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from numpy import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import numpy as np
from minepy import MINE
import pandas as pd

#
def mic(x, y):
    mine = MINE()
    mine.compute_score(x, y)
    return mine.mic()


def ML(F, C, r):
    """
    F: Features in ndarray format of size (s, k)
    C: Label in ndarray format, containing only 0 and 1, of size (s,)
    r: A pre-set irrelevency threshold
    """
    s, k = F.shape
    micFC = [-1 for _ in range(k)]
    Subset = [-1 for _ in range(k)]

    numSubset = 0  # [0, numSubset) contains the selected features

    for i in range(k):
        micFC[i] = mic(F[:, i].flatten(), C.flatten())


        if micFC[i] >= r:
            Subset[numSubset] = i
            numSubset += 1

    Subset = Subset[0:numSubset]
    Subset.sort(key=lambda x: micFC[x], reverse=True)

    mask = [True for _ in range(numSubset)]
    for e in range(numSubset):
        if mask[e]:
            for q in range(e + 1, numSubset):
                if mask[q] and mic(F[:, Subset[e]], F[:, Subset[q]]) >= micFC[Subset[q]]:
                    mask[q] = False
    FReduce = F[:, np.array(Subset)[mask]]
    sel = np.array(Subset)[mask]


    return FReduce,sel

def evaluation(X, y):
    """
    Get the greatest accuracy of SVM, NBayes, Dtree, NN
    """
    y = y.astype('int')
    kf = KFold(n_splits=5)
    mAuc = []
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        auc1 = np.mean(SVC().fit(X_train, y_train).predict(X_test) == y_test)
        auc2 = np.mean(GaussianNB().fit(X_train, y_train).predict(X_test) == y_test)
        auc3 = np.mean(DecisionTreeClassifier().fit(X_train, y_train).predict(X_test) == y_test)
        auc4 = np.mean(KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train).predict(X_test) == y_test)
        mAuc.append(max(auc1, auc2, auc3, auc4))
    return np.array(mAuc).mean()


def evalu(X1, label_data):

    X1 = pd.DataFrame(X1)
    label_data = pd.DataFrame(label_data)

    X_train, X_test, y_train, y_test = train_test_split(X1, label_data, test_size=0.3, random_state=50)


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)

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

        acc_x_run = []
        f1_x_run = []
        auc_x_run = []
        acc_r_run = []
        f1_r_run = []
        auc_r_run = []
        acc_s_run = []
        f1_s_run = []
        auc_s_run = []
    for i, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
        train_x1, train_y = X_train.iloc[train_index], y_train.iloc[train_index]
        test_x1, test_y = X_train.iloc[test_index], y_train.iloc[test_index]

        gnb1r = RandomForestClassifier(random_state=0)
        gnb1r.fit(train_x1, train_y)
        y_pred1r = gnb1r.predict(test_x1)
        y_pro1r = gnb1r.predict_proba(test_x1)[:, 1]
        acc_r.append(accuracy_score(test_y, y_pred1r))
        f1_r.append(f1_score(test_y, y_pred1r))
        auc_r.append(roc_auc_score(test_y, y_pro1r))

        print('RandomForest :')
        print('acc:', sum(acc_r) / len(acc_r))
        print('f1:', sum(f1_r) / len(f1_r))
        print('auc:', sum(auc_r) / len(auc_r))
        print('aucmax:', max(auc_r))
        return max(auc_r)


if __name__ == "__main__":
    path = '../result/'
    # data preparation
    data = pd.read_csv(path + 'netensa.csv', index_col=0).T.values
    col = pd.read_csv(path + 'netensa.csv', index_col=0).T.columns


    label = pd.read_csv(path + 'label.csv', index_col=0).values

    Decision,sel = ML(data, label, 0.1)
    X1_col = pd.DataFrame(col)
    result_df = pd.DataFrame(columns=['species'])
    for i in range(sel.shape[0]):
        index = sel[i]
        print(X1_col.iloc[index][0])

        value_to_save = X1_col.iloc[index][0] 
        result_df = result_df.append({'species': value_to_save}, ignore_index=True)
    result_df.to_csv(path + 'ensa_sel.csv', index=False)

    Auc = evalu(Decision, label)
    print('FOne.shape: ',Decision.shape)
    print(f'mAuc: {Auc}')

    