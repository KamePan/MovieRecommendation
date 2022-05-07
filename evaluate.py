import math
import time
import pandas as pd
import numpy as np
from neo4j import GraphDatabase

import transH
from transE import TransE
import database
import sklearn.model_selection
import matplotlib.pyplot as plt
from transH import TransH


def evaluate():
    """
    将 ratings 数据集分为 train 训练集及 test 测试集两个集合，比例一般为 3：7
    获取物品的预测评分和物品的预测评分
    $$ 如何检测获得均方根误差 RMSE
    预测用户的评分时，
    对于每个用户的评分，假设对 n 个电影有评分，则将 ceil(0.7 * n) 作为用户评分训练集，将 floor(0.3 * n) 作为测试集
    ***** 物品的语义相似度提前计算存起来，不然到时候训练时间过长 *****
    将语义相似度存放在 ([movie1, movie2], similarity) 的字典数据结构中
    由于是要测试电影评分，我们要预测的电影只需要是测试集中电影的评分就可以
    具体到我们的算法中，
    由于要预测算法的效率，因此需要统计对所有用户评分的效果。
    ***** 如果要提升训练的效率，就需要提前算出每个电影与其他电影的协同过滤相似度 *****
    MATCH Movie - RATED - User - RATED - Movie
    通过评分加权得到 Movie 的协同过滤物品相似度，与语义相同将这个相似度存放在 ([movie1, movie2], similarity) 的字典数据结构中

    for user in users: # 对于每个用户
        获得 user 的评分向量，按照比例将评分按照 3:7 分为测试集和训练集
        for test_movie in test_set:
            for movie_train in train_set:
                从字典中得到 (test_movie, other_movie) 的协同过滤相似度和语义相似度
                将协同过滤物品相似度与语义相似度加权融合得到融合相似度
                predict_rating += Similarity_train_test * Rating_Movie_train
            并进行归一化
        计算均方根误差：RMSE = SUM(SQRT((movie_test_rating - predict_rating)^2))
    """
    '''获取电影集合及用户评分矩阵（能够优化电影集合，比如计算协同过滤相似度时只需要获取有评分的电影集合）'''
    cf_movie_set, semantic_movie_set, user_rating_dict = query_movie_and_user_rating_dict()
    '''计算协同过滤相似度矩阵'''
    cf_similarity_dict = calculate_cf_similarity_table(cf_movie_set)
    '''计算语义相似度矩阵'''
    semantic_transE_similarity_dict = calculate_semantic_similarity_table(semantic_movie_set, trans_type=1)
    semantic_transH_similarity_dict = calculate_semantic_similarity_table(semantic_movie_set, trans_type=2)
    '''计算总体均方根误差RMSE'''
    transE_RMSE = calculate_overall_RMSE(cf_similarity_dict, semantic_transE_similarity_dict, user_rating_dict)
    transH_RMSE = calculate_overall_RMSE(cf_similarity_dict, semantic_transH_similarity_dict, user_rating_dict)
    # origin_list_arr, predict_list_arr = calculate_overall_RMSE(cf_similarity_dict, semantic_transE_similarity_dict, user_rating_dict)
    alpha_arr = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.plot(alpha_arr, transE_RMSE)
    plt.plot(alpha_arr, transH_RMSE)
    # RMSE = [0, 0, 0, 0, 0, 0]
    # for index in range(0, 6):
    #     origin_list = origin_list_arr[index]
    #     predict_list = predict_list_arr[index]
    #     RMSE[index] = get_rmse(origin_list, predict_list)
    # plt.plot(alpha_arr, RMSE)
    plt.xlabel('alpha')  # x轴标题
    plt.ylabel('RMSE')   # y轴标题
    # plt.legend(['TransE+Item-CF'])
    plt.legend(['TransE+Item-CF', 'TransH+Item-CF '])
    # plt.title('Use Euclidean Metric To Measure CF Item Similarity')
    plt.title('Use Cos Distance To Measure CF Item Similarity')
    plt.show()


def calculate_overall_RMSE(cf_similarity_dict, semantic_similarity_dict, user_rating_dict):
    """"""
    '''由于 RMSE 是根据所有用户测试集预测评分中的偏差值建立的'''
    '''尝试使用建立预测评分列表和真实评分列表计算 RMSE'''
    '''发现没什么区别，我之前算的均方误差值是正确的...'''
    # RMSE = 0
    sample_num = 0
    origin_predict_txt = open("./result/out_origin_predict.txt", 'w')
    predict_list = [list(), list(), list(), list(), list(), list()]
    origin_list = [list(), list(), list(), list(), list(), list()]
    '''绘制评估指标图所用数组'''
    '''混合比例 alpha 数组'''
    '''RMSE 计算结果存储数组'''
    alpha_arr = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    RMSE_arr = [0, 0, 0, 0, 0, 0]
    for user_id in user_rating_dict.keys():
        '''这里希望对每个用户都计算其评分过的电影中被划分到测试集中电影的预测评分'''
        user_rating_movie_list = list(user_rating_dict[user_id].keys())
        '''
        随机函数将用户评分过的电影随机 3:7 分为测试集、训练集
        对于测试集中的电影，根据训练集中的电影融合相似度与其加权得到预测评分
        '''
        '''评分不超过一部电影的用户不计入统计范围内'''
        if len(user_rating_movie_list) <= 1:
            continue
        train_set, test_set = sklearn.model_selection.train_test_split(user_rating_movie_list, test_size=0.3)

        for test_movie in test_set:
            # predict_test_movie_rating = 0
            # similarity_sum = 0
            '''绘制评估指标图所用数组'''
            similarity_sum_arr = [0, 0, 0, 0, 0, 0]
            predict_test_movie_rating_arr = [0, 0, 0, 0, 0, 0]
            for train_movie in train_set:
                '''考虑两部电影可能由于没有共同评分者，不存在协同过滤相似度'''
                cf_similarity = 0
                if (test_movie, train_movie) in cf_similarity_dict.keys():
                    cf_similarity = cf_similarity_dict[(test_movie, train_movie)]
                if (train_movie, test_movie) in cf_similarity_dict.keys():
                    cf_similarity = cf_similarity_dict[(train_movie, test_movie)]
                '''两部电影一定存在语义相似度'''
                semantic_similarity = 0
                if (test_movie, train_movie) in semantic_similarity_dict.keys():
                    semantic_similarity = semantic_similarity_dict[(test_movie, train_movie)]
                if (train_movie, test_movie) in semantic_similarity_dict.keys():
                    semantic_similarity = semantic_similarity_dict[(train_movie, test_movie)]
                '''设置相似度融合权重 alpha'''

                '''这里能对 alpha 直接做处理吗'''
                for index, alpha in enumerate(alpha_arr):
                    merge_similarity = alpha * semantic_similarity + (1 - alpha) * cf_similarity
                    predict_test_movie_rating_arr[index] += merge_similarity * user_rating_dict[user_id][train_movie]
                    similarity_sum_arr[index] += abs(merge_similarity)
            origin_test_movie_rating = user_rating_dict[user_id][test_movie]
            origin_predict_txt.write(f"{origin_test_movie_rating}")
            '''这里通过 test_movie 的 predict_rating 与 origin_rating 计算 RMSE'''
            for index in range(0, 6):
                predict_test_movie_rating_arr[index] /= similarity_sum_arr[index]
                '''为每个用户的每个测试数据计算 RMSE'''
                RMSE_arr[index] += (origin_test_movie_rating - predict_test_movie_rating_arr[index]) ** 2
                '''这里将不同 alpha 的预测评分和真实评分加入列表中'''
                predict_list[index].append(predict_test_movie_rating_arr[index])
                origin_list[index].append(origin_test_movie_rating)
                origin_predict_txt.write(f", {predict_test_movie_rating_arr[index]}")
                # alpha = 0.3
                # merge_similarity = alpha * semantic_similarity + (1 - alpha) * cf_similarity
                # predict_test_movie_rating += merge_similarity * user_rating_dict[user_id][train_movie]
                # similarity_sum += abs(merge_similarity)
            # '''归一化得到最终预测评分'''
            # predict_test_movie_rating /= similarity_sum
            # origin_test_movie_rating = user_rating_dict[user_id][test_movie]
            # '''为每个用户的每个测试数据计算 RMSE'''
            # RMSE += (origin_test_movie_rating - predict_test_movie_rating) ** 2
            origin_predict_txt.write("\n")
            sample_num += 1
    # RMSE = np.sqrt(RMSE / sample_num)
    # print(f'推荐算法评价指标 RMSE 值：{RMSE}')
    print(f"sample_num: {sample_num}")
    for index in range(0, 6):
        RMSE_arr[index] = np.sqrt(RMSE_arr[index] / sample_num)
    return RMSE_arr


def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None


def query_movie_and_user_rating_dict():
    """
    cf_movie_set: 仅统计有用户评分的电影
    semantic_movie_set: 统计所有电影
    """
    movie_dataframe = pd.read_csv('./dataset/out_movie_new.csv')
    semantic_movie_set = set()
    for index, row in movie_dataframe.iterrows():
        movie_title = row['title']
        semantic_movie_set.add(movie_title)
    print(f"电影总数: {len(semantic_movie_set)}")

    rating_dataframe = pd.read_csv('./dataset/out_grade_new.csv')
    user_rating_dict = dict()
    cf_movie_set = set()
    rating_cnt = 0
    for index, row in rating_dataframe.iterrows():
        user_id = row['user_id']
        title = row['title']
        rating = row['grade']
        cf_movie_set.add(title)
        '''还未导入过该 user 的任何评分信息'''
        if user_id not in user_rating_dict.keys():
            user_rating_dict[user_id] = dict()
        user_rating_dict[user_id][title] = rating
        rating_cnt += 1
    print(f"用户数量: {len(user_rating_dict.keys())}")
    print(f"评分数量: {rating_cnt}")
    print(f"评分电影数量: {len(cf_movie_set)}")
    return cf_movie_set, semantic_movie_set, user_rating_dict


def calculate_semantic_similarity_table(movie_title_set, trans_type=1):
    print("计算语义相似度矩阵...")
    start_time_semantic = time.time()
    semantic_similarity_dict = dict()
    '''这里载入的数据是某个 User 的数据，但算出的 entity 嵌入向量则是基于所有实体的'''
    entity_set, relation_set, triple_list = database.query_entity_relation_set_and_triple_list()
    transH.relation_tph_hpt_loader(triple_list)
    movie_id_set, movie_dict = database.query_movie_dict()
    trans = None
    if trans_type == 1:
        trans = TransE(entity_set, relation_set, triple_list, embedding_dim=50, learning_rate=0.01, margin=1)
        trans.emb_initialize()
        trans.train(epochs=10, nbatches=400)
    elif trans_type == 2:
        trans = TransH(entity_set, relation_set, triple_list, embedding_dim=50, learning_rate=0.01, margin=1)
        trans.emb_initialize()
        trans.train(epochs=200, nbatches=400)

    '''将 (movie_id, embedding) 字典转为 (title, embedding) 字典'''
    title_embedding_dict = dict()
    # out_semantic_similarity = open("./dataset/out_semantic_similarity.txt", 'w')
    for entity_id in trans.entity.keys():
        '''排除不是电影的实体'''
        if entity_id in movie_dict.keys():
            title_embedding_dict[movie_dict[entity_id]] = trans.entity[entity_id]
    for movie1 in movie_title_set:
        for movie2 in movie_title_set:
            if movie1 == movie2:
                continue
            movie1_vec = title_embedding_dict[movie1]
            movie2_vec = title_embedding_dict[movie2]
            '''语义相似度计算采用欧式距离'''
            sub_vec = movie1_vec - movie2_vec
            semantic_similarity = 1 / (1 + np.sqrt(np.inner(sub_vec, sub_vec)))
            semantic_keys = semantic_similarity_dict.keys()
            if (movie1, movie2) not in semantic_keys and (movie2, movie1) not in semantic_keys:
                semantic_similarity_dict[(movie1, movie2)] = semantic_similarity
                # out_semantic_similarity.write(f"{(movie1, movie2)}:  {semantic_similarity} \n")
            # similarity = np.inner(movie1_vec, movie2_vec) / (
            #         np.sqrt(np.inner(movie1_vec, movie1_vec)) * np.sqrt(np.inner(movie2_vec, movie2_vec)))
    end_time_semantic = time.time()
    print(f"计算语义相似度矩阵时间: {str(end_time_semantic - start_time_semantic)} s")
    return semantic_similarity_dict


def calculate_cf_similarity_table(movie_set):
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "pan151312"), max_connection_lifetime=200)
    session = driver.session()
    print("计算协同过滤相似度矩阵...")
    start_time_cf = time.time()
    cf_similarity_dict = dict()
    for movie in movie_set:
        '''如果没有人同时看过这两部电影，那么协同过滤相似度为 0，这时候就让语义相似度决定其混合相似度'''
        '''这里由两层 movie 循环优化为单层循环，只算出与和 movie 有相同评分人的电影的相似度，减少了冗余'''
        '''鉴于计算时间仍然很长，优化方向：1. 只使用有用户评分的电影集遍历 2. 尝试不采用图数据库计算'''
        '''这种方式仍然会将每个电影的相似度都计算两遍，可以通过构造评分矩阵的方式来优化'''
        '''欧式距离计算电影协同过滤相似度：SQRT(SUM((r1.grading - r2.grading)^2)) AS sim'''
        '''1 / (1 + np.sqrt(np.sum((movie1_vec - movie2_vec) ** 2)))'''
        '''余弦距离计算电影协同过滤相似度：SUM(r1.grading * r2.grading) / (SQRT(SUM(r1.grading^2)) * SQRT(SUM(r2.grading^2))) AS sim'''
        query_result = session.run(f"""
            MATCH (m1:Movie{{title:\"{movie}\"}})-[r1:RATED]-(u:User)-[r2:RATED]-(m2:Movie)
            WITH
                m1.title AS title1, m2.title AS title2, 
                COUNT(u) AS users_common,
                1 / (1 + SQRT(SUM((r1.grading - r2.grading)^2))) AS sim
            WHERE users_common > 0 
            RETURN title1, title2, sim 
        """)
        query_data = query_result.data()
        for record in query_data:
            movie1 = record['title1']
            movie2 = record['title2']
            cf_similarity = record['sim']
            cf_keys = cf_similarity_dict.keys()
            if (movie1, movie2) not in cf_keys and (movie2, movie1) not in cf_keys:
                cf_similarity_dict[(movie1, movie2)] = cf_similarity
    end_time_cf = time.time()
    print(f"计算协同过滤相似度矩阵时间：{str(end_time_cf - start_time_cf)} s")
    return cf_similarity_dict


if __name__ == '__main__':
    evaluate()