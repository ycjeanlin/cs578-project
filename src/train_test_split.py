import pandas as pd
from collections import defaultdict
import math
import sys
import codecs

def validate_train_test_split(train_file, test_file):
    train_data = defaultdict(lambda: defaultdict(float))
    test_data = defaultdict(lambda: defaultdict(float))
    with codecs.open(train_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            cols = line.strip().split('\t')
            train_data[int(cols[0])][int(cols[1])] = float(cols[2])

    with codecs.open(test_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            cols = line.strip().split('\t')
            test_data[int(cols[0])][int(cols[1])] = float(cols[2])

    for u in test_data:
        for i in test_data[u]:
            if i in train_data[u]:
                raise Exception('testing entry is in training data')

    print('Test data pass')


if __name__ == "__main__":
    '''
    Command: python3 train_test_split.py [path_to_raw_data] [k-fold]
    Example: python3 train_test_split.py ../data/ml-100k/u.data 5
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
        n_train = n - n_test
        test_data = test_data.append(u_log[n_train:])
        #print(n_train, len(u_log[:n_train]), len(u_log[n_train:]))


        for i in range(k):
            val_data[i] = val_data[i].append(u_log[math.floor(n_train*i/k):math.floor(n_train*(i+1)/k)])
            train_data[i] = train_data[i].append(u_log[:math.floor(n_train*i/k)])
            train_data[i] = train_data[i].append(u_log[math.floor(n_train*(i+1)/k):n_train])

    test_data.to_csv(path_or_buf = "../data/testing/test.dat", sep="\t", header = False, index=False)

    for i in range(k):
        val_data[i].to_csv(path_or_buf = "../data/validation/validation_"+str(i+1)+".dat", sep="\t", header = False, index=False)
        train_data[i].to_csv(path_or_buf = "../data/training/train_"+str(i+1)+".dat", sep="\t", header = False, index=False)

    # n_train = 10-2
    # for  i in range(k):
    #     print("----")
    #     print(list(range(math.floor(n_train*i/k), math.floor(n_train*(i+1)/k))))
    #     print(list(range(math.floor(n_train*i/k))))
    #     print(list(range(math.floor(n_train * (i+1) / k), n_train)))

    #validate_train_test_split('../data/training/train.dat', '../data/testing/test.dat')
