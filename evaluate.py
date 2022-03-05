import time
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from transE import TransE, data_loader
import sklearn.model_selection


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
            for other_movie in movie_set:
                从字典中得到 (test_movie, other_movie) 的协同过滤相似度和语义相似度
                将协同过滤物品相似度与语义相似度加权融合得到融合相似度
            for movie_train in train_set:
                predict_rating += Similarity_train_test * Rating_Movie_train
            并进行归一化
        计算均方根误差：RMSE = SUM(SQRT((movie_test_rating - predict_rating)^2))
    """
    '''获取电影集合'''
    movie_set, user_set = query_movie_and_user_set()
    '''计算协同过滤相似度矩阵'''
    cf_similarity_dict = calculate_cf_similarity_table(movie_set)
    '''计算语义相似度矩阵'''
    semantic_similarity_dict = calculate_semantic_similarity_table(movie_set)
    ''''''
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "pan151312"), max_connection_lifetime=200)
    session = driver.session()
    '''计算总体均方根误差RMSE'''
    RMSE = 0
    sample_num = 0
    for user_id in user_set:
        query_result = session.run(f"""
                MATCH (u:User{{id:{user_id}}})-[r:RATED]-(m:Movie) RETURN r, m
            """)
        result_data = query_result.data()
        movie_rating_dict = dict()
        for rating_record in result_data:
            rating = rating_record['r']['grading']
            movie = rating_record['m']['title']
            movie_rating_dict[movie] = rating
        '''
        随机函数将用户评分过的电影随机 3:7 分为测试集、训练集(未实现)
        对于测试集中的电影，根据训练集中的电影融合相似度与其加权得到预测评分
        '''
        test_set = set()
        train_set = set()
        for test_movie in test_set:
            predict_test_movie_rating = 0
            similarity_sum = 0
            for train_movie in train_set:
                '''考虑两部电影可能由于没有共同评分者，不存在协同过滤相似度'''
                cf_similarity = 0
                if (test_movie, train_movie) in cf_similarity_dict.keys():
                    cf_similarity = cf_similarity_dict[test_movie]
                if (train_movie, test_movie) in cf_similarity_dict.keys():
                    cf_similarity = cf_similarity_dict[test_movie]
                semantic_similarity = 0
                if (test_movie, train_movie) in semantic_similarity_dict.keys():
                    semantic_similarity = semantic_similarity_dict[test_movie]
                if (train_movie, test_movie) in semantic_similarity_dict.keys():
                    semantic_similarity = semantic_similarity_dict[test_movie]
                '''设置相似度融合权重 alpha'''
                alpha = 0.3
                merge_similarity = alpha * semantic_similarity + (1 - alpha) * cf_similarity
                predict_test_movie_rating += merge_similarity * movie_rating_dict[train_movie]
                similarity_sum += merge_similarity
            '''归一化得到最终预测评分'''
            predict_test_movie_rating /= similarity_sum
            origin_test_movie_rating = movie_rating_dict[test_movie]
            '''为每个用户的每个测试数据计算 RMSE'''
            RMSE += (origin_test_movie_rating - predict_test_movie_rating) ** 2
            sample_num += 1
    RMSE = np.sqrt(RMSE / sample_num)
    print(f'推荐算法评价指标 RMSE 值：{RMSE}')


def query_movie_and_user_set():
    movie_dataframe = pd.read_csv('./dataset/out_movie_new.csv')
    movie_set = set()
    for index, row in movie_dataframe.iterrows():
        movie_title = row['title']
        movie_set.add(movie_title)
    print(f"电影数量: {len(movie_set)}")

    user_dataframe = pd.read_csv('./dataset/out_user_new.csv')
    user_set = set()
    for index, row in user_dataframe.iterrows():
        user_id = row['user_id']
        user_set.add(user_id)
    print(f"用户数量: {len(user_set)}")
    return movie_set, user_set


def calculate_semantic_similarity_table(movie_title_set):
    print("计算语义相似度矩阵...")
    start_time_semantic = time.time()
    semantic_similarity_dict = dict()
    entity_set, relation_set, triple_list, movie_id_set, movie_dict, current_user_grade_dict = data_loader(440)
    trans = TransE(entity_set, relation_set, triple_list, embedding_dim=20, learning_rate=0.01, margin=1, L1=True)
    trans.emb_initialize()
    trans.train(epochs=10)
    '''将 (movie_id, embedding) 字典转为 (title, embedding) 字典'''
    title_embedding_dict = dict()
    '''entity 不一定是 movie'''
    for entity_id in trans.entity.keys():
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
        query_result = session.run(f"""
            MATCH (m1:Movie{{title:\"{movie}\"}})-[r1:RATED]-(u:User)-[r2:RATED]-(m2:Movie)
            WITH
                m1.title AS title1, m2.title AS title2, 
                COUNT(u) AS users_common,
                SUM(r1.grading * r2.grading) / (SQRT(SUM(r1.grading^2)) * SQRT(SUM(r2.grading^2))) AS sim
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
