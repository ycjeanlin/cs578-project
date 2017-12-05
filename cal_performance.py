from collections import defaultdict
import codecs
import sys
import pandas as pd


def read_test_data(file_name):
    # read testing data
    u_l = defaultdict(lambda: defaultdict(float))
    i_l = defaultdict(lambda: defaultdict(float))
    with codecs.open(file_name, 'r', encoding='utf-8') as fr:
        for line in fr:
            cols = line.strip().split('\t')
            u_l[int(cols[0])][int(cols[1])] = float(cols[2])
            i_l[int(cols[1])][int(cols[0])] = float(cols[2])
    return u_l, i_l


def read_recomm_data(file_name):
    # read recommendation list
    u_l = {}
    with codecs.open(file_name, 'r', encoding='utf-8') as fr:
        for line in fr:
            user, items = line.strip().split(':')
            u_l[int(user)] = set()
            for i in items.split(','):
                u_l[int(user)].add(int(i))
    return u_l


def cal_avg_precision(recom_lists, test_list):
    precisions = []
    for u in recom_lists:
        hit = 0
        for i in recom_lists[u]:
            if i in test_list:
                hit += 1
        precision = float(hit/len( recom_lists[u]))
        precisions.append(precision)

    return sum(precisions) / float(len(precisions))


def cal_avg_recall(recom_lists, test_list):
    recalls = []
    for u in test_list:
        hit = 0
        for i in test_list[u]:
            if i in recom_lists:
                hit += 1
        recall = float(hit / len(test_list[u]))
        recalls.append(recall)

    return sum(recalls) / float(len(recalls))


def cal_accuracy(recom_lists, test_list):
    hit = 0
    for u in test_list:
        for i in test_list[u]:
            if i in recom_lists:
                hit += 1
                break

    accuracy = float(hit / len(test_list))

    return accuracy

def export_result(out_file, result):
    result.to_csv(path_or_buf=out_file)

if __name__ == '__main__':

    min_k = int(sys.argv[1])
    max_k = int(sys.argv[2])
    delta_k = int(sys.argv[3])

    user_lists, item_lists = read_test_data('./data/testing/test.dat')
    exp_result = pd.DataFrame(index=range(min_k, max_k+1, delta_k), columns=['Precision', 'Recall', 'Accuracy'])
    for k in range(min_k, max_k+1, delta_k):
        recomm_lists = read_recomm_data('list_1_100_{}.txt'.format(k))
        avg_precision = cal_avg_precision(recomm_lists, user_lists)
        avg_recall = cal_avg_recall(recomm_lists, user_lists)
        accuracy = cal_accuracy(recomm_lists, user_lists)
        exp_result.loc[k, 'Precision'] = avg_precision
        exp_result.loc[k, 'Recall'] = avg_recall
        exp_result.loc[k, 'Accuracy'] = accuracy

    export_result('performance.csv', exp_result)