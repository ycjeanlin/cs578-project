import pandas as pd
from collections import defaultdict
import math
import sys


if __name__ == "__main__":
    '''
    Command: python train_test_split.py [path_to_data] [k-fold]
    Example: python train_test_split.py ./data/ml-100k/u.data 5
    '''
    raw_data_file = sys.argv[1]
    k = int(sys.argv[2])

    df = pd.read_csv(raw_data_file, sep='\t',header=None)
    user_set = set(df[0])
    train_data = defaultdict(pd.DataFrame)
    val_data = defaultdict(pd.DataFrame)
    test_data = pd.DataFrame()

    for u in user_set:
        u_log = df.loc[df[0] == u]
        n = len(u_log)
        n_test = math.floor(n*0.2)
        test_data = test_data.append(u_log.loc[-n_test:])
        n_train = n-n_test

        for i in range(k):
            val_data[i] = val_data[i].append(u_log[math.floor(n_train*i/k):math.floor(n_train*(i+1)/k)])
            train_data[i] = train_data[i].append(u_log[:math.floor(n_train*i/k)])
            train_data[i] = train_data[i].append(u_log[math.floor(n_train*(i+1)/k):])

    test_data.to_csv(path_or_buf = "test.dat", sep="\t", header = False, index=False)

    for i in range(k):
        val_data[i].to_csv(path_or_buf = "validation_"+str(i+1)+".dat", sep="\t", header = False, index=False)
        train_data[i].to_csv(path_or_buf = "train_"+str(i+1)+".dat", sep="\t", header = False, index=False)

    # n_train = 10-2
    # for  i in range(k):
    #     print("----")
    #     print(list(range(math.floor(n_train*i/k), math.floor(n_train*(i+1)/k))))
    #     print(list(range(math.floor(n_train*i/k))))
    #     print(list(range(math.floor(n_train * (i+1) / k), n_train)))
