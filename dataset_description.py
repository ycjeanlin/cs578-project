# coding: utf-8

# Attention: fig.show() will interrupt running of this code. 
# 			Gentlely colse the figure poped up to let code continue running.
#			Or just comment those.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

sns.set_context("talk")
sns.set_style("whitegrid")
rcParams['patch.force_edgecolor'] = True
# get_ipython().magic('matplotlib inline')


# # Load Dataset
# There is a total of 99,999 ratings in this dataset. For every row, first two 
# entries are the user id and movie id, which can be used to identify user and 
# movie. The third entry is the rating, in this dataset, all ratings are integers
# in range 1 to 5. The last entry is a time stamp, which is unix seconds since 1/1/1970 UTC.

ratings =  pd.read_csv('ml-100k/u.data', sep='\t',  header=0, 
                       names=['userId', 'movieId', 'rating','timestamp'], engine='python').astype(int)
print(ratings.head())


# # User Description
# We have a total of 943 users. Each of them rated at least 20 movies and at 
# most 737 movies. The mean number of rated movie for users is 106 and standard
# deviation is around 100. It is a long-tailed distribution, which means most people 
# rated 100 or less movies, and only few people rated a lot.
print(ratings['userId'].value_counts().describe())


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
plt.suptitle('User Rating Distribution',fontsize=20)
sns.distplot(ratings['userId'].value_counts(), kde=False, ax=ax1)
sns.distplot(ratings['userId'].value_counts(), ax=ax2)
plt.show()

# # Movie Description
# We have a total of 1682 users. They have been rated at least 1 time and at most 583
# times. Mean value of number of ratings is around 60 but standard deviation is
# around 80. Most movies get 10 ratings or less.
print(ratings['movieId'].value_counts().describe())

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
plt.suptitle('Movie Rated Distribution',fontsize=20)
sns.distplot(ratings['movieId'].value_counts(), kde=False, ax=ax1)
sns.distplot(ratings['movieId'].value_counts(), ax=ax2)
plt.show()


# # Ratings Description
# We have a total of 99,999 ratings in range 1 to 5, involve only integers. 4 is most 
# occured in the ratings, and 3 is the second most. Over a half of ratings are 3 or 4. 
# The mean value of ratings is 3.5.
print(ratings['rating'].describe())

fig, (ax1) = plt.subplots(ncols=1, figsize=(6,5))
plt.suptitle('Rating Distribution',fontsize=20)
sns.distplot(ratings['rating'], kde=False, ax=ax1)
plt.show()
