import codecs
import sys
import pickle

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
            model = ItemCF('./train_{}.dat'.format(i))
            model.training('./model_{}.dat'.format(i))
            export_cf_model(model, './model_{}.dat'.format(i))

    for i in range(1, num_train + 1):
        model = load_cf_model('./model_{}.dat'.format(i))
        user_lists, item_lists = model.read_data('./train_{}.dat'.format(i))
        for n in range(min_n, max_n+1 , delta_n):
            similar_items = model.find_similar_items(n)

            rank_list = {}
            for u in user_lists:
                logger.debug("User " + str(u))
                rank_list[u] = model.recommend_items(user_lists[u], similar_items)

            for k in range(min_k, max_k+1, delta_k):
                top_k_list = {}
                for u in user_lists:
                    top_k_list[u] =  [str(rank_list[u][i][0]) for i in range(k)]

                export_rank_list(top_k_list, './list_{}_{}_{}.txt'.format(i, n, k))
                logger.info("Done... Check " + './list_{}_{}_{}.txt'.format(i, n, k))

