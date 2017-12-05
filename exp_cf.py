import codecs
import sys
import pickle
import math
import csv
import pandas as pd

from ItemCF import ItemCF

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def export_rank_list(rank_list, out_rank_list):
    with codecs.open(out_rank_list, 'w', encoding='utf-8') as fw:
        for u in rank_list:
            fw.write(str(u) + ":")
            fw.write((',').join(rank_list[u]))
            fw.write('\n')

def export_cf_model(cf_model, obj_file):
    with codecs.open(obj_file, 'wb') as fw:
        pickle.dump(cf_model, fw)

def load_cf_model(obj_file):
    with codecs.open(obj_file, 'rb') as fr:
        cf_model = pickle.load(fr)
    return cf_model

def cal_rmse(ratings, user_list):
    error = 0.0
    n = 0.0
    for u in ratings:
        for i in ratings[u]:
            n += 1
            if i in user_list[u]:
                error += (user_list[u][i]-ratings[u][i]) * (user_list[u][i]-ratings[u][i])
            else:
                error += (user_list[u][i] - 4.0) * (user_list[u][i] - 4.0)
    return math.sqrt(error/n)

def export_rmse(out_file, result):
    result.to_csv(path_or_buf=out_file)

if __name__ == "__main__":
    '''
    python3 exp_cf.py train 3 10 15 5 5 5 1
    '''

    status = sys.argv[1]
    num_train = int(sys.argv[2])
    min_n = int(sys.argv[3])
    max_n = int(sys.argv[4])
    delta_n = int(sys.argv[5])
    min_k = int(sys.argv[6])
    max_k = int(sys.argv[7])
    delta_k = int(sys.argv[8])

    if status == "train":
        for i in range(1, num_train + 1):
            model = ItemCF('./data/training/train.dat'.format(i))
            model.training('./model_item_cf.obj'.format(i))
            export_cf_model(model, './model_item_cf.obj'.format(i))


    exp_result = pd.DataFrame(index=range(min_n, max_n+1 , delta_n), columns=range(1, num_train + 1))
    for i in range(1, num_train + 1):
        model = load_cf_model('./model_item_cf.obj'.format(i))
        user_lists, item_lists = model.read_data('./data/testing/test.dat'.format(i))
        rmse_n = []
        for n in range(min_n, max_n+1 , delta_n):
            print('Number of Neighbors', n)
            similar_items = model.find_similar_items(n)

            rank_list = {}
            ratings = {}

            for u in user_lists:
                logger.debug("User " + str(u))
                ratings[u] = model.predict_ratings(user_lists[u], similar_items)
                rank_list[u] = model.recommend_items(ratings[u], user_lists[u])

            #exp_result.loc[n, i] = cal_rmse(ratings, user_lists)

            for k in range(min_k, max_k + 1, delta_k):
                top_k_list = {}
                for u in user_lists:
                    top_k_list[u] = [str(rank_list[u][i][0]) for i in range(k)]

                export_rank_list(top_k_list, './list_{}_{}_{}.txt'.format(i, n, k))


    #export_rmse('./rmse.csv', exp_result)




    logger.info("Done... ")

