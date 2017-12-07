import codecs
from collections import defaultdict
import math
import sys
import pickle
import operator

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserCF:

    def __init__(self, train_f):
        self.train_file = train_f
        self.user_similarity = {}

    def read_data(self, file_name):
        u_l = defaultdict(lambda : defaultdict(float))
        i_l = defaultdict(lambda : defaultdict(float))
        with codecs.open(file_name, 'r', encoding='utf-8') as fr:
            for line in fr:
                cols = line.strip().split('\t')
                u_l[int(cols[0])][int(cols[1])] = float(cols[2])
                i_l[int(cols[1])][int(cols[0])] = float(cols[2])
        return u_l, i_l

    def _cal_user_similarity(self, user_lists):

        index = 0
        logger.info('Number of users: ' + str(len(user_lists)))
        sqrt_u = {}

        logger.info('calculate square root....')
        for u in user_lists:
            sqrt_u[u] = 0.0
            for i in user_lists[u]:
                sqrt_u[u] += user_lists[u][i] * user_lists[u][i]
            sqrt_u[u] = math.sqrt(sqrt_u[u])

        logger.info("calculate similarity...")
        for u in user_lists:
            index += 1
            if index % 10 == 0:
                percentage = float(index) / len(user_lists) * 100
                logger.info(('{:.2f}% complete...').format(percentage))
            # Loop through the columns for each column
            for v in user_lists:
                if u not in self.user_similarity:
                    self.user_similarity[u] = {}

                if v not in self.user_similarity:
                    self.user_similarity[v] = {}

                if u < v:
                    inner_product = 0.0
                    if len(user_lists[u]) > 0 and len(user_lists[v]) > 0:
                        for i in user_lists[u]:
                            if i in user_lists[v]:
                                inner_product += user_lists[u][i] * user_lists[v][i]

                        self.user_similarity[u][v] = inner_product / sqrt_u[u] / sqrt_u[v]
                        self.user_similarity[v][u] = self.user_similarity[u][v]
                    else:
                        self.user_similarity[u][v] = 0
                        self.user_similarity[v][u] = 0
                elif u == v:
                    self.user_similarity[u][v] = 1.0


    def training(self, out_obj_file):
        logger.info("Training...")
        user_lists, item_lists = self.read_data(self.train_file)
        self._cal_user_similarity(user_lists)


    def find_similar_users(self, n):
        logger.debug("Finding similar neighbors...")
        similar_neighbors = defaultdict(lambda: defaultdict(float))
        for i in self.user_similarity:
            sort_sim_list = sorted(self.user_similarity[i].items(), key=operator.itemgetter(1),reverse=True)
            for j in range(n+1):
                if i != sort_sim_list[j][0]:
                    similar_neighbors[i][sort_sim_list[j][0]] = sort_sim_list[j][1]

        return similar_neighbors

    def predict_ratings(self, u, user_lists, similar_neighbors):
        rating_i = defaultdict(int)
        normalization = defaultdict(int)
        for v in similar_neighbors[u]:
            for i in user_lists[v]:
                rating_i[i] += similar_neighbors[u][v] * user_lists[v][i]
                normalization[i] += similar_neighbors[u][v]

        for i in rating_i:
            rating_i[i] /= (normalization[i] + 1)

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

    model = UserCF(train_file)
    model.training(out_obj_file)

    similar_users = model.find_similar_users(n)

    user_lists, item_lists = model.read_data(train_file)
    top_k_list = {}
    for u in user_lists:
        logger.debug("User " + str(u))
        top_k_list[u] = model.recommend_items(user_lists[u], similar_users)

    logger.info("Export recommendation list...")
    export_rank_list(top_k_list, out_rank_list)

    logger.info("Done... Check " + out_rank_list)

