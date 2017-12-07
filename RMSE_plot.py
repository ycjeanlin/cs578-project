
# coding: utf-8

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

sns.set_context("talk")
sns.set_style("whitegrid")
rcParams['patch.force_edgecolor'] = True
# get_ipython().magic('matplotlib inline')

df = pd.read_csv('exp_result/mf_rmse_step1.csv')
df = df.set_index('Unnamed: 0')
df = df.transpose()
df.columns.name = 'features (m)'
# df.head()

plt.suptitle('RMSE vs. Top-n Similar Users ',fontsize=14)
# plt.title('(Test Step of 1, m from 1 to 24)', fontsize=10)
sns.tsplot(data=df.as_matrix(), time=df.columns, value='RMSE')
plt.savefig('plot/mf_rmse_step1.png')

