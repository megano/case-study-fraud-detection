from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GDBC
from sklearn.ensemble import AdaBoostClassifier as ADR
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, accuracy_score, recall_score, f1_score, precision_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import json


def load_data(filename):
    with open('./data/' + filename) as data_file:
        data = json.load(data_file)
    df = pd.DataFrame.from_dict(data)
    df['eur'] = df.currency.map(lambda x: 1 if (x == 'eur') else 0)
    df['gbp'] = df.currency.map(lambda x: 1 if (x == 'gbp') else 0)
    df['ach'] = df.payout_type.map(lambda x: 1 if x == 'ACH' else 0)
    df['check'] = df.payout_type.map(lambda x: 1 if x == 'CHECK' else 0)
    df['missing_payment'] = (1 - df.ach - df.check)
    df['dict_elements'] = df.previous_payouts.map(lambda x: len(x))
    return df.fillna(0)


def prepare_data(df, y_name='fraud'):
    y = df.pop(y_name).values
    X = df.values
    return X, y


# def compare_cross_val_score(model_list, X, y):
#     results = []
#     for model in model_list:
#         scores = cross_val_score(
#             model, X, y, cv=5, scoring=make_scorer(recall_score))
#         mean_score = scores.mean()
#         results.append((model.__class__.__name__, mean_score))
#     # print results
#     return results


# def plot_compare_cross_val_score(model_list, X, y):
#     results = compare_cross_val_score(model_list, X, y)
#     scores = [tup[1] for tup in results]
#     labels = [tup[0] for tup in results]
#     width = 0.45
#     fig, ax = plt.subplots()
#     ind = np.arange(len(labels))
#     ax.bar(ind, scores, width, color='g')
#     ax.set_xticks(ind + width / 2)
#     ax.set_xticklabels(labels)
#     ax.set_ylim(0, 1.25)
#     ax.set_ylabel('Cross Val - Recall score')
#     plt.show()
#     pass


def grid_search(model_type, params, X, y):
    model = GridSearchCV(model_type, params,
                         scoring=make_scorer(recall_score), cv=5, n_jobs=-1)
    model.fit(X, y)
    print model.best_params_, model.best_score_
    return model.best_estimator_, model.best_params_, model.best_score_


def scores_for_best_model(best_est, X, y):
    best_est.fit(X, y)
    predictions = best_est.predict(X)
    recall = recall_score(y, predictions)
    precision = precision_score(y, predictions)
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions)
    print "F1 score: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, Accuracy: {:.2f}".format(f1, precision, recall, accuracy)
    return f1, precision, recall, accuracy


def plot_roc_curve(best_est, X, y):
    best_est.fit(X, y)
    predictions = best_est.predict_proba(X)
    # pred_logistic = LR().fit(X, y).predict_proba(X)
    # pred_randomforest = RF().fit(X,y).predict_proba(X)

    # print pred_randomforest[:10]
    # find roc numbers
    fpr, tpr, thresholds = roc_curve(y, predictions[:, 1], pos_label=1)
    # fpr2, tpr2, thresholds2 = roc_curve(y, pred_logistic[:, 1], pos_label=1)
    # fpr3, tpr3, thresholds3 = roc_curve(y, pred_randomforest[:,1], pos_label=1)

    # plot roc curve
    x_line = np.linspace(0, 1, 70)
    y_line = x_line
    plt.plot(fpr, tpr, label='RF Classifier (best model)', color='b')
    # plt.plot(fpr2, tpr2, label='Logistic Regression', color='black')
    #plt.plot(fpr3, tpr3, label='Random Forest Classifier', color='g')
    plt.plot(x_line, y_line, color='r')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    pass


# def test(filename, best_est_fitted):
#     X_test, y_test = prepare_data(filename)
#     predictions = best_est_fitted.predict(X_test)
#     #import ipdb; ipdb.set_trace()
#     recall = recall_score(y_test, predictions)
#     precision = precision_score(y_test, predictions)
#     accuracy = accuracy_score(y_test, predictions)
#     f1 = f1_score(y_test, predictions)
#     print "F1 score: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, Accuracy: {:.3f}".format(f1, precision, recall, accuracy)
#     return f1, precision, recall, accuracy


if __name__ == '__main__':
    df = load_data('data.json')
    nick_col = ['fraud', 'eur', 'gbp', 'ach', 'check', 'missing_payment', 'dict_elements', 'gts',
                'has_logo', 'user_type', 'delivery_method', 'org_facebook', 'org_twitter', 'has_analytics']

    df_nick = df.loc[:, nick_col]
    X, y = prepare_data(df_nick)
    # compare_cross_val_score([RF(), GDBC()], X, y)

    # grid search RF
    params_rf = {'n_estimators': [50, 100, 500], 'max_depth': [
        1, 2, 5, 10], 'min_samples_split': [2, 4]}
    gs_rf = grid_search(RF(), params_rf, X, y)
    scores_for_best_model(gs_rf[0], X, y)
    #     F1 score: 0.87, Precision: 0.96, Recall: 0.79, Accuracy: 0.98
    # (0.86963906581740968,
    #  0.96421845574387943,
    #  0.79195668986852286,
    #  0.97858687312547954)

    # plot roc curve for RF
    plot_roc_curve(gs_rf[0], X, y)
    plt.legend()
    plt.show()
