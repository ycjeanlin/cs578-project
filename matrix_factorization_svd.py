# coding: utf-8

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

training_files = ['data/training/train_1.dat', 'data/training/train_2.dat',
                  'data/training/train_3.dat', 'data/training/train_4.dat', 'data/training/train_5.dat']
validation_files = ['data/validation/validation_1.dat', 'data/validation/validation_2.dat',
                   'data/validation/validation_3.dat', 'data/validation/validation_4.dat', 'data/validation/validation_5.dat']
out_file = 'exp_result/mf_rmse_step5.csv' # change out file name
# result
rmse = {}

for m in range(25, 76, 5):  # change testing m and step
    print("m = ", m)
    rmse[m] = [0 for _ in range(5)]
    for case in range(5):
        ratings = pd.read_csv(training_files[case], sep='\t',  header=0, names=[
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

        # Validation
        validation_rating = pd.read_csv(validation_files[case], sep='\t',  header=0,
                                        names=['userId', 'movieId', 'rating', 'timestamp'], engine='python').astype(int)

        def predict(row):
            try:
                return preds_df.loc[row['userId'] - 1, row['movieId'] - 1]
            except:
                return 4 # return median as default predict
        validation_rating['pred rating'] = validation_rating.apply(
            lambda row:  predict(row), axis=1)
        validation_rating.head()

        rmse[m][case] = np.sqrt(mean_squared_error(
            validation_rating['rating'], validation_rating['pred rating']))
        print(rmse[m][case])

rmse_df = pd.DataFrame.from_dict(rmse, orient='index')
rmse_df.to_csv(out_file)
