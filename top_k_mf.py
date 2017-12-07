# coding: utf-8

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

training_file = 'data/training/train.dat'
testing_file = 'data/testing/test.dat'
out_file = 'exp_result/mf_performance.csv'
m = 6
precisions = {}
recalls = {}
hits = {}

for k in [5, 10, 15, 20, 25, 30]:
    print(k)
    ratings = pd.read_csv(training_file, sep='\t',  header=0, names=[
                        'userId', 'movieId', 'rating', 'timestamp'], engine='python')

    R_df = pd.pivot_table(ratings, values='rating',
                        index='userId', columns='movieId').fillna(0)
                        
    # normalized matrix
    R = R_df.as_matrix()
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    # SVD decomposition
    U, sigma, Vt = svds(R_demeaned, k=m)
    sigma = np.diag(sigma)

    # Prediction
    all_user_predicted_ratings = np.dot(
        np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(
        all_user_predicted_ratings, columns=R_df.columns)

    # top k recommendations for users
    top_k_list = {}
    for user in preds_df.index.unique():
        rated_movie = ratings[ratings['userId'] == (user)]['movieId']
        pred_rating = preds_df[[x for x in preds_df.columns.values if x not in rated_movie.values]].loc[user].sort_values(ascending=False)
        top_k_list[user] =pred_rating.index.values.tolist()[:k]
    #top_k_list

    # Validation, not sure how to do yet
    validation_rating = pd.read_csv(testing_file, sep='\t',  header=0,
                                    names=['userId', 'movieId', 'rating', 'timestamp'], engine='python').astype(int)
    validation_list = validation_rating.groupby('userId')['movieId'].apply(lambda x: x.tolist()).to_dict()

    precision = []
    recall = []
    hit = []
    for user in top_k_list.keys():
        corr = 0
        for movie in top_k_list[user]:
            if movie in validation_list[user+1]:
                corr = corr + 1
        precision.append(corr/len(top_k_list[user]))
        recall.append(corr/len(validation_list[user+1]))
        if corr > 0:
            hit.append(1)
        else:
            hit.append(0)

    precisions[k]= np.mean(precision)
    recalls[k] = np.mean(recall)
    hits[k] = np.mean(hit)

summary_df = pd.DataFrame()
summary_df['Precision'] = pd.Series(precisions)
summary_df['Recall'] = pd.Series(recalls)
summary_df['Accuracy'] = pd.Series(hits)
summary_df.to_csv(out_file)