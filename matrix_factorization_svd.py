
# coding: utf-8

import pandas as pd
import numpy as np

training_files = ['ml-100k/u1.base', 'ml-100k/u2.base', 'ml-100k/u3.base', 'ml-100k/u4.base']
validation_file = 'ml-100k/u5.base'
out_file = 'mf-k.csv'
k=5

frames = []
for training_file in training_files:
    frames.append(pd.read_csv(training_file, sep='\t',  header=0, 
                       names=['userId', 'movieId', 'rating','timestamp'], engine='python').astype(int))
ratings = pd.concat(frames)
ratings.head()

# R_df = ratings.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
R_df = pd.pivot_table(ratings,values='rating',index='userId',columns='movieId').fillna(0)
R_df.head()

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

top_k_df = pd.DataFrame.from_dict(top_k_list, orient='index')
top_k_df.head()

# export to file
top_k_df.to_csv(out_file)


# Validation, not sure how to do yet
validation_rating = pd.read_csv(validation_file, sep='\t',  header=0, 
                       names=['userId', 'movieId', 'rating','timestamp'], engine='python').astype(int)
validation_rating.head()

def predict(row):
    return preds_df.loc[row['userId'] - 1, row['movieId']]
validation_rating['pred rating'] = validation_rating.apply (lambda row:  predict(row),axis=1)
validation_rating.head()

from sklearn.metrics import mean_squared_error
mean_squared_error(validation_rating['rating'], validation_rating['pred rating'])

