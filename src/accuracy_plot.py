import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

sns.set_context("talk")
sns.set_style("whitegrid")
rcParams['patch.force_edgecolor'] = True
# get_ipython().magic('matplotlib inline')

mf_df = pd.read_csv('exp_result/mf_performance.csv')
mf_df = mf_df.set_index('Unnamed: 0')
# mf_df = mf_df.transpose()

item_cf_df = pd.read_csv('exp_result/item_cf_performance.csv')
item_cf_df = item_cf_df.set_index('Unnamed: 0')
# item_cf_df = item_cf_df.transpose()

user_cf_df = pd.read_csv('exp_result/user_cf_performance.csv')
user_cf_df = user_cf_df.set_index('Unnamed: 0')
# user_cf_df = user_cf_df.transpose()

summary_df = pd.DataFrame(columns=['Accuracy', 'Method'])
summary_df = pd.concat([summary_df, mf_df]).fillna('Matrix Factorization')
summary_df = pd.concat([summary_df, item_cf_df]).fillna('Item-based Collaborative Filtering')
summary_df = pd.concat([summary_df, user_cf_df]).fillna('User-based Collaborative Filtering')
summary_df['k'] = summary_df.index

sns.factorplot(x="k", y="Accuracy", hue="Method", data=summary_df, legend_out=False, size=5, aspect=1.4)
plt.suptitle('Accuracy vs. k',fontsize=20)
plt.savefig('plot/accuracy.png')

sns.factorplot(x="k", y="Precision", hue="Method", data=summary_df, legend_out=False, size=5, aspect=1.4)
plt.suptitle('Precision vs. k',fontsize=20)
plt.savefig('plot/precision.png')

sns.factorplot(x="k", y="Recall", hue="Method", data=summary_df, legend_out=False, size=5, aspect=1.4)
plt.suptitle('Recall vs. k',fontsize=20)
plt.savefig('plot/recall.png')
