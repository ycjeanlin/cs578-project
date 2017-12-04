# coding: utf-8

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

training_files = ['data/training/train_1.dat', 'data/training/train_2.dat',
                  'data/training/train_3.dat', 'data/training/train_4.dat', 'data/training/train_5.dat']
validation_files = ['data/validation/validation_1.dat', 'data/validation/validation_2.dat',
                   'data/validation/validation_3.dat', 'data/validation/validation_4.dat', 'data/validation/validation_5.dat']

m = 6

for k in [5, 10, 15, 20, 25, 30]:
    for case in range(5):
        ratings = pd.read_csv(training_files[case], sep='\t',  header=0, names=[
                            'userId', 'movieId', 'rating', 'timestamp'], engine='python')
        # ratings.head()

        R_df = pd.pivot_table(ratings, values='rating',
                            index='userId', columns='movieId').fillna(0)
        # print(R_df.head())
        # print(R_df.shape)

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
        # print(preds_df.shape)

        # top k recommendations for users
        top_k_list = {}
        for user in preds_df.index.unique():
            rated_movie = ratings[ratings['userId'] == (user)]['movieId']
            pred_rating = preds_df[[x for x in preds_df.columns.values if x not in rated_movie.values]].loc[user].sort_values(ascending=False)
            top_k_list[user] =pred_rating.index.values.tolist()[:k]
        #top_k_list

        # Validation, not sure how to do yet
        validation_rating = pd.read_csv(validation_files[case], sep='\t',  header=0,
                                        names=['userId', 'movieId', 'rating', 'timestamp'], engine='python').astype(int)
        validation_list = validation_rating.groupby('userId')['movieId'].apply(lambda x: x.tolist()).to_dict()

        precision = []
        recall = []
        hit = []

        for user in top_k_list.keys():
            hits = 0
            for movie in top_k_list[user]:
                if movie in validation_list[user+1]:
                    hits = hits + 1
            precision.append(hits/len(top_k_list[user]))
            recall.append(hits/len(validation_list[user+1]))
            if hits > 0:
                hit.append(1)
            else:
                hit.append(0)

        print("k:" + str(k) + "\tcase:" + str(case))
        print(np.mean(precision))
        print(np.mean(recall))
        print(np.mean(hit))

'''
result

k:5     case:0
0.0984093319194
0.0522221934088
0.406150583245
k:5     case:1
0.101378579003
0.0544095200625
0.428419936373
k:5     case:2
0.100954400848
0.0537461793674
0.415694591729
k:5     case:3
0.100954400848
0.0516353926086
0.41145281018
k:5     case:4
0.0979851537646
0.0497256216045
0.397667020148
k:10    case:0
0.0933191940615
0.0943817078178
0.604453870626
k:10    case:1
0.0951219512195
0.0981403255229
0.640509013786
k:10    case:2
0.0952279957582
0.0967512765255
0.632025450689
k:10    case:3
0.096394485684
0.0957499964125
0.627783669141
k:10    case:4
0.098197242842
0.0969110321548
0.638388123012
k:15    case:0
0.090491339696
0.132276499468
0.734888653234
k:15    case:1
0.090491339696
0.135160961498
0.764581124072
k:15    case:2
0.0911983032874
0.134156510743
0.746553552492
k:15    case:3
0.0933898904206
0.134030815933
0.75079533404
k:15    case:4
0.0963591375044
0.137310824831
0.773064687169
k:20    case:0
0.0879109225875
0.169224815644
0.810180275716
k:20    case:1
0.0881230116649
0.168942775075
0.833510074231
k:20    case:2
0.088441145281
0.169174360066
0.817603393425
k:20    case:3
0.0895546129374
0.165636292188
0.819724284199
k:20    case:4
0.0935312831389
0.173616894149
0.852598091198
k:25    case:0
0.084835630965
0.201743297261
0.863202545069
k:25    case:1
0.083520678685
0.195643455536
0.872746553552
k:25    case:2
0.0869565217391
0.203480266848
0.862142099682
k:25    case:3
0.0862778366914
0.196062877748
0.858960763521
k:25    case:4
0.0918769883351
0.20932308173
0.893955461294
k:30    case:0
0.0821491693178
0.230716871752
0.889713679745
k:30    case:1
0.0812654648286
0.22402812968
0.903499469777
k:30    case:2
0.0848709791446
0.231337322933
0.901378579003
k:30    case:3
0.0836691410392
0.225127153273
0.897136797455
k:30    case:4
0.0885825379993
0.236028933203
0.916224814422
'''