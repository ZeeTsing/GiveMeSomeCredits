import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score,accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

def model_fit_train_score_skf(model, X, y, kfold=5):
    '''This function takes in three arguments:model (model object), X,y
    It will be splitted by stratified k fold algo
    The data will be fitted using the model passed in by the user
    It returns the fitted model object and lists of Accuracy score as well as F1 score and AUC (area under curve)'''
    skf = StratifiedKFold(n_splits=kfold)
    results_dict = defaultdict()
    predict = []
    predict_prob = []
    Accuracy = []
    F1 = []
    AUC = []
    y_vals = []

    for train_index, test_index in skf.split(X, y):
        # get current split
        x_train, x_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        # fit model with latest train set
        model.fit(x_train, y_train)
        # calculate predictions
        y_pred = model.predict(x_val)
        predictions = model.predict_proba(x_val)
        predict.append(y_pred)
        predict_prob.append(predictions[:, 1])
        Accuracy.append(accuracy_score(y_true=y_val, y_pred=y_pred))
        F1.append(f1_score(y_true=y_val, y_pred=y_pred))
        AUC.append(roc_auc_score(y_val, predictions[:, 1]))
        y_vals.append(y_val)

    results_dict['y_val'] = y_vals
    results_dict['predictions'] = predict
    results_dict['predict_proba'] = predict_prob
    results_dict['Accuracy_mean'] = np.mean(Accuracy)
    results_dict['F1_mean'] = np.mean(F1)
    results_dict['AUC_mean'] = np.mean(AUC)
    results_dict['Accuracy_std'] = np.std(Accuracy)
    results_dict['F1_std'] = np.std(F1)
    results_dict['AUC_std'] = np.std(AUC)

    return model, results_dict

def plot_ROC(y_true, y_proba, AUC, figsize = (7,5), color = 'darkturquoise', title='ROC Curve'):
    '''Helper function to plot ROC graph'''
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    #set size
    plt.figure(figsize=figsize)
    #plot
    plt.plot(fpr, tpr,lw=2,c=color,label = f"AUC: {AUC:.2f}")
    #adjustments
    plt.plot([0,1],[0,1],c='grey',ls='--')
    plt.legend()
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title);