import codecs
from collections import defaultdict
import pandas as pd
import math
import sys
import pickle
import operator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ItemCF:

    def __init__(self, train_f):
        self.train_file = train_f
        self.item_similarity = {}

    def read_data(self, file_name):
        u_l = defaultdict(lambda : defaultdict(float))
        i_l = defaultdict(lambda : defaultdict(float))
        with codecs.open(file_name, 'r', encoding='utf-8') as fr:
            for line in fr:
                cols = line.strip().split('\t')
                u_l[int(cols[0])][int(cols[1])] = float(cols[2])
                i_l[int(cols[1])][int(cols[0])] = float(cols[2])
        return u_l, i_l

    def _cal_item_similarity(self, item_lists):

        index = 0
        logger.info('Number of items: ' + str(len(item_lists)))
        sqrt_i = {}

        logger.info('calculate square root....')
        for i in item_lists:
            sqrt_i[i] = 0.0
            for u in item_lists[i]:
                sqrt_i[i] += item_lists[i][u] * item_lists[i][u]
            sqrt_i[i] = math.sqrt(sqrt_i[i])

        logger.info("calculate similarity...")
        for i in item_lists:
            index += 1
            if index % 10 == 0:
                percentage = float(index) / len(item_lists) * 100
                logger.info(('{:.2f}% complete...').format(percentage))
            # Loop through the columns for each column
            for j in item_lists:
                if i not in self.item_similarity:
                    self.item_similarity[i] = {}

                if j not in self.item_similarity:
                    self.item_similarity[j] = {}

                if i < j:
                    inner_product = 0.0
                    if len(item_lists[i]) > 0 and len(item_lists[j]) > 0:
                        for u in item_lists[i]:
                            if u in item_lists[j]:
                                inner_product += item_lists[i][u] * item_lists[j][u]

                        self.item_similarity[i][j] = inner_product / sqrt_i[i] / sqrt_i[j]
                        self.item_similarity[j][i] = inner_product / sqrt_i[i] / sqrt_i[j]
                    else:
                        self.item_similarity[i][j] = 0
                        self.item_similarity[j][i] = 0
                elif i == j:
                    self.item_similarity[i][j] = 1.0


    def training(self, out_obj_file):
        logger.info("Training...")
        user_lists, item_lists = self.read_data(self.train_file)
        self._cal_item_similarity(item_lists)


    def find_similar_items(self, n):
        logger.debug("Finding similar neighbors...")
        similar_neighbors = defaultdict(lambda: defaultdict(float))
        for i in self.item_similarity:
            sort_sim_list = sorted(self.item_similarity[i].items(), key=operator.itemgetter(1),reverse=True)
            for j in range(n+1):
                if i != sort_sim_list[j][0]:
                    similar_neighbors[i][sort_sim_list[j][0]] = sort_sim_list[j][1]

        return similar_neighbors

    def predict_ratings(self, item_list, similar_neighbors):
        rating_i = defaultdict(int)
        normalization = defaultdict(int)
        for i in item_list:
            for j in similar_neighbors[i]:
                rating_i[j] += similar_neighbors[i][j] * item_list[i]
                normalization[j] += similar_neighbors[i][j]

        for j in rating_i:
            rating_i[j] /= (normalization[j] + 1)

        return rating_i

    def recommend_items(self, rating_i, train_user_list):
        for i in train_user_list:
            rating_i[i] = 0
        rank_list = sorted(rating_i.items(), key=operator.itemgetter(1),reverse=True)

        return rank_list


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
    status = sys.argv[1]
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    train_file = sys.argv[4]
    validation_file = sys.argv[5]
    out_obj_file = sys.argv[6]
    out_rank_list = sys.argv[7]

    model = ItemCF(train_file)
    model.training(out_obj_file)

    similar_items = model.find_similar_items(n)

    user_lists, item_lists = model.read_data(train_file)
    top_k_list = {}
    for u in user_lists:
        logger.debug("User " + str(u))
        top_k_list[u] = model.recommend_items(user_lists[u], similar_items)

    logger.info("Export recommendation list...")
    export_rank_list(top_k_list, out_rank_list)

    logger.info("Done... Check " + out_rank_list)

