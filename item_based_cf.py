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


def read_data(file_name):
    u_l = defaultdict(lambda : defaultdict(float))
    i_l = defaultdict(lambda : defaultdict(float))
    item_set = set()
    with codecs.open(file_name, 'r', encoding='utf-8') as fr:
        for line in fr:
            cols = line.strip().split('\t')
            u_l[int(cols[0])][int(cols[1])] = float(cols[2])
            i_l[int(cols[1])][int(cols[0])] = float(cols[2])

    return u_l, i_l

def cal_item_similarity(item_list):
    item_similarity = pd.DataFrame(index=item_lists.keys(), columns=item_lists.keys())

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
            if i < j:
                inner_product = 0.0
                if len(item_lists[i]) > 0 and len(item_lists[j]) > 0:
                    for u in item_lists[i]:
                        if u in item_lists[j]:
                            inner_product += item_lists[i][u] * item_lists[j][u]

                    item_similarity.ix[i, j] = inner_product / sqrt_i[i] / sqrt_i[j]
                    item_similarity.ix[j, i] = inner_product / sqrt_i[i] / sqrt_i[j]
                else:
                    item_similarity.ix[i, j] = 0
                    item_similarity.ix[j, i] = 0
            elif i == j:
                item_similarity.ix[i, j] = 1.0
    return item_similarity

def export_item_similarity(similarity_matrix, obj_file):
    with codecs.open(obj_file, 'wb') as fw:
        pickle.dump(similarity_matrix, fw)

def load_item_similarity(obj_file):
    with codecs.open(obj_file, 'rb') as fr:
        similarity_matrix = pickle.load(fr)
    return similarity_matrix

def find_similar_items(n, similarity_matrix):
    logger.debug("Finding similar neighbors...")
    similar_neighbors = defaultdict(lambda: defaultdict(float))
    for i in similarity_matrix:
        sort_sim_list = sorted(similarity_matrix[i].items(), key=operator.itemgetter(1),reverse=True)
        for j in range(n+1):
            if i != sort_sim_list[j][0]:
                similar_neighbors[i][sort_sim_list[j][0]] = sort_sim_list[j][1]

    return similar_neighbors

def recommend_items(k, item_list, similar_neighbors):
    score_j = defaultdict(int)
    normalization =  defaultdict(int)

    logger.info("Calculating recommended scores...")
    for i in item_list:
        for j in similar_neighbors[i]:
            score_j[j] += similar_neighbors[i][j] * item_list[i]
            normalization[j] += similar_neighbors[i][j]

    for j in score_j:
        score_j[j] /= (normalization[j]+1)

    rank_list = sorted(score_j.items(), key=operator.itemgetter(1),reverse=True)

    return [str(rank_list[i][0]) for i in range(k)]


def export_rank_list(rank_list, out_rank_list):
    with codecs.open(out_rank_list, 'w', encoding='utf-8') as fw:
        for u in rank_list:
            fw.write(str(u) + ":")
            fw.write((',').join(rank_list[u]))
            fw.write('\n')


if __name__ == "__main__":
    status = sys.argv[1]
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    train_file = sys.argv[4]
    validation_file = sys.argv[5]
    out_obj_file = sys.argv[6]
    out_rank_list = sys.argv[7]

    if status == 'train':
        logger.info("Training...")
        user_lists, item_lists = read_data(train_file)
        item_similarity = cal_item_similarity(item_lists)
        export_item_similarity(item_similarity, out_obj_file)

    logger.info("Validating...")
    user_lists, item_lists = read_data(validation_file)
    item_similarity = load_item_similarity(out_obj_file)
    similar_items = find_similar_items(n, item_similarity)

    top_k_list = {}
    for u in user_lists:
        logger.debug("User " + str(u))
        top_k_list[u] = recommend_items(k, user_lists[u], similar_items)

    logger.info("Export recommendation list...")
    export_rank_list(top_k_list, out_rank_list)

    logger.info("Done... Check " + out_rank_list)

