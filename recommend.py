import copy
import math
import random
import time
import database
import numpy as np
import matplotlib.pyplot as plt

from transE import TransE
from transH import TransH

import cf

def query_semantic_recommendation_top_k_by_user_id(user_id, recommend_num=10, a=1.0):
    start = time.time()
    entity_set, relation_set, triple_list = database.query_entity_relation_set_and_triple_list()
    movie_set, movie_dict = database.query_movie_dict()
    current_user_rating_dict = database.query_rating_dict_by_user_id(user_id)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    start_translate_train = time.time()
    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=20, learning_rate=0.01, margin=1, L1=True)
    transE.emb_initialize()
    transE.train(epochs=50)
    end_translate_train = time.time()
    print("TransE train time: " + str(end_translate_train - start_translate_train) + "s")
    # 得到协同过滤的物品相似度表
    start_predict = time.time()
    predict_list = []
    trans_similarity_table = dict()
    for movie1 in movie_set:  # 遍历所有电影
        # 不需要求出每个电影间的相似度，只需要求出当前用户看过的电影与其他电影的相似度就可以
        # 其他电影的预测评分则通过用户评分过的电影评分和相似度进行相乘加权即可
        # 这里的电影相似度矩阵如何构造呢
        # 考虑将不将相似度存起来，而是将相似度直接与用户评分相乘
        if movie1 in current_user_rating_dict.keys():  # 滤除用户看过的电影
            continue
        movie1_vec = transE.entity[movie1]
        movie1_predict = 0
        similarity_sum = 0
        for movie2 in current_user_rating_dict.keys():  # 遍历用户看过的电影
            movie2_vec = transE.entity[movie2]
            '''采用欧式距离计算相似度'''
            sub_vec = movie1_vec - movie2_vec
            similarity = 1 / (1 + np.sqrt(np.inner(sub_vec, sub_vec)))
            # similarity = np.inner(movie1_vec, movie2_vec) / (
            #             np.sqrt(np.inner(movie1_vec, movie1_vec)) * np.sqrt(np.inner(movie2_vec, movie2_vec)))
            movie1_predict += similarity * current_user_rating_dict[movie2]
            similarity_sum += abs(similarity)
        # 除去相似度之和进行归一化
        if similarity_sum == 0:
            continue
        movie1_predict = movie1_predict / similarity_sum
        predict_list.append([movie_dict[movie1], movie1_predict])
    predict_list = sorted(predict_list, key=(lambda movie: movie[1]), reverse=True)[:recommend_num]

    for predict_movie in predict_list:
        predict_movie_title = predict_movie[0]
        recommender_num, genres = database.query_movie_recommmender_num_and_genres_by_title(predict_movie_title)
        predict_movie.append(recommender_num)
        predict_movie.append(genres)
    end_predict = time.time()
    print("build transE predict list: " + str(end_predict - start_predict) + "s")
    end = time.time()
    print("query_recommendation_top_k_by_user_id: " + str(end - start) + "s")
    return predict_list


def query_merged_recommendation_top_k_by_user_id(user_id, recommend_num=10, a=1.0):
    start = time.time()
    print("load data...")
    entity_set, relation_set, triple_list = database.query_entity_relation_set_and_triple_list()
    movie_set, movie_dict = database.query_movie_dict()
    current_user_rating_dict = database.query_rating_dict_by_user_id(user_id)
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    start_translate_train = time.time()
    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=20, learning_rate=0.01, margin=1, L1=True)
    transE.emb_initialize()
    transE.train(epochs=50)
    end_translate_train = time.time()
    print("TransE train time: " + str(end_translate_train - start_translate_train) + "s")

    '''得到协同过滤的物品相似度表'''
    current_user_rated_movie_similarity_table = cf.query_current_user_rated_movie_similarity_table(user_id)

    start_predict = time.time()
    predict_list = []
    for movie1 in movie_set:  # 遍历所有电影
        # 不需要求出每个电影间的相似度，只需要求出当前用户看过的电影与其他电影的相似度就可以
        # 其他电影的预测评分则通过用户评分过的电影评分和相似度进行相乘加权即可
        # 这里的电影相似度矩阵如何构造呢
        # 考虑将不将相似度存起来，而是将相似度直接与用户评分相乘
        if movie1 in current_user_rating_dict.keys():  # 滤除用户看过的电影
            continue
        movie1_vec = transE.entity[movie1]
        movie1_predict = 0
        similarity_sum = 0
        for movie2 in current_user_rating_dict.keys():  # 遍历用户看过的电影
            movie2_vec = transE.entity[movie2]
            # 采用欧式距离并将其规约到 (0,1] 之间
            # 这里物品相似度加权为融合相似度，但一部分电影的无协同过滤相似度，因为没有给其评分的人，这种情况下直接用语义相似度作为融合相似度
            # 即将该电影认定为新电影，若语义关联度高，则评分高。
            similarity = np.inner(movie1_vec, movie2_vec) / (
                    np.sqrt(np.inner(movie1_vec, movie1_vec)) * np.sqrt(np.inner(movie2_vec, movie2_vec)))
            # similarity = 1 / (1 + np.sqrt(np.sum((movie1_vec - movie2_vec) ** 2)))
            cf_similarity = similarity
            if movie1 in current_user_rated_movie_similarity_table[movie2].keys():
                cf_similarity = current_user_rated_movie_similarity_table[movie2][movie1]
            merged_similarity = a * similarity + (1 - a) * cf_similarity
            # 未评分电影 movie1 和评分电影 movie2 相似度与 movie1 评分
            movie1_predict += merged_similarity * current_user_rating_dict[movie2]
            similarity_sum += abs(merged_similarity)
        # 除去相似度之和进行归一化
        movie1_predict = movie1_predict / similarity_sum
        predict_list.append([movie_dict[movie1], movie1_predict])
    predict_list = sorted(predict_list, key=(lambda movie: movie[1]), reverse=True)[:recommend_num]
    # 放在这里做查询就能够只做 recommend_num 次数的查询
    for predict_movie in predict_list:
        predict_movie_title = predict_movie[0]
        recommender_num, genres = database.query_movie_recommmender_num_and_genres_by_title(predict_movie_title)
        predict_movie.append(recommender_num)
        predict_movie.append(genres)
    end_predict = time.time()
    print("build transE predict list: " + str(end_predict - start_predict) + "s")
    end = time.time()
    print("query_recommendation_top_k_by_user_id: " + str(end - start) + "s")
    return predict_list
    # 能否提取出相似度矩阵呢
    # 这里要提取出相似度矩阵其实是可以的，但是比较麻烦，能不能不构造相似度矩阵，直接在计算出相似度的时候进行合并呢
    # 但是这样就需要同时计算出语义相似度和协同过滤相似度
    # 计算语义相似度的时候能通过先前已经计算得到的向量，计算出用户未看过电影A与看过电影B的相似度
    # 在这里能否计算出电影A和电影B协同过滤的相似度呢
    # 当然可以啊，因为 cf 的实现就是先计算相似度存到 Neo4j 里面，那么只要取出以 A 和 B 为首尾的 similarity 关系就可以了
    # 不对，因为 cf 实现是基于用户的协同过滤，只会计算用户的相似度，怎么可能把用户的相似度矩阵和物品的相似度矩阵加权融合
    # 现在的策略就是 1.用基于物品的协同过滤，将电影间的相似度矩阵加权相加 2.将两个算出的 TopK 进行融合
