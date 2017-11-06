# coding: utf-8

import pandas as pd
import numpy as np
import csv


training_file = 'ml-100k/u1.base'
out_file = 'mf-k.csv'
k=5


ratings =  pd.read_csv(training_file, sep='\t',  header=0, 
                       names=['userId', 'movieId', 'rating','timestamp'], engine='python').astype(int)
# ratings.head()


R_df = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
# R_df.head()


# normalized matrix
R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)


# SVD decomposition
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)
sigma = np.diag(sigma)


# Prediction
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)


# top k recommendations for users
top_k_list = {}
for user in preds_df.index.unique():
    rated_movie = ratings[ratings['userId'] == (user)]['movieId']
    pred_rating = preds_df[[x for x in preds_df.columns.values if x not in rated_movie.values]].loc[user].sort_values(ascending=False)
    top_k_list[user] =pred_rating.index.values.tolist()[:k]

# output
top_k_df = pd.DataFrame.from_dict(top_k_list, orient='index')
top_k_df.to_csv(out_file)

