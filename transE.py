from neo4j import GraphDatabase
from py2neo import Graph, Relationship, Node
import pandas as pd
import codecs
import copy
import math
import random
import time
import database as db


import numpy as np

entity2id = {}
relation2id = {}
loss_ls = []

movie_set = set()
movie_dict = dict()
current_user_grade_dict = dict()

# data_loader 从 FB15K 数据集文本中提取 entity 和 relation 以及三元组数据。这里三元组数据中关系有很多，而我们的电影数据集仅有几个关系，
# 那么我们电影数据集对应到这里的三元组是 (User, Movie, RATED) 对应 (h, t, r)，一个问题是例如 Genre 的关系和实体需要放进来吗？
# TransE 计算时仅考虑三元组和实体，那么如果我们能够将实体和其向量很好对应起来，那么将所有的实体和关系（不管是电影、用户还是类型）都使用
# TransE 表示为向量形式，基于相似电影表示出的向量相近这一命题，则我们将电影对应的向量计算相似度，就可以得到推荐结果。

# 1. 查询图数据库中的关系得到三元组信息
# 2. 构造所有实体和关系集合
# 3. 构造所有关系的三元组 (User, Movie, RATED), (Movie, Genre, HAS_GENRE), (Movie, Productor, PRODUCED_BY),
#    (Movie, keyword, HAS_KEYWORD)
# 4. 将实体集、关系集、三元组传入 TransE 训练，即为实体和关系随机生成 k 维向量，并建立实体到向量的对应关系（即字典形式），
#    通过三元组进行向量训练，得到实体和关系的向量
# 5. 如何将训练出的向量与推荐评分相结合呢
# 6. 由于推荐系统为特定 ID 用户推荐商品，则可以与基于物品的协同过滤相同，构造（物品，物品相似度矩阵），进而根据用户对物品的评分和相似度进行加权评分


def data_loader(user_id):
    uri = "neo4j://localhost:7687"
    graph = Graph(uri, auth=("neo4j", "pan151312"))

    entity_set = set()
    relation_set = set()
    triple_list = []

    grades = graph.run(f"""
        MATCH (User{{id:{user_id}}})-[r:RATED]-(m:Movie) RETURN m, r LIMIT 500
    """)
    for grade_record in grades:
        movie = grade_record["m"]
        grade = grade_record["r"]
        current_user_grade_dict[str(movie.identity)] = grade["grading"]
    # print(current_user_grade_dict)

    all_relations = graph.run(f"""
        MATCH relation=()-->() RETURN relation LIMIT 10000
    """)
    for relation_record in all_relations:
        relation = relation_record["relation"]
        start_entity = relation.start_node
        end_entity = relation.end_node
        relation_type = list(relation.types())[0]
        entity_set.add(str(start_entity.identity))
        entity_set.add(str(end_entity.identity))
        relation_set.add(relation_type)
        triple_list.append([str(start_entity.identity), str(end_entity.identity), relation_type])

        if start_entity.has_label('Movie'):
            movie_set.add(str(start_entity.identity))
            movie_dict[str(start_entity.identity)] = start_entity["title"]
        if end_entity.has_label('Movie'):
            movie_set.add(str(end_entity.identity))
            movie_dict[str(end_entity.identity)] = end_entity["title"]
    df = pd.DataFrame(triple_list)
    # print(df)

    return entity_set, relation_set, triple_list


def distanceL2(h, r, t):
    # 为方便求梯度，去掉sqrt
    return np.sum(np.square(h + r - t))


def distanceL1(h, r, t):
    return np.sum(np.fabs(h + r - t))


class TransE:
    def __init__(self, entity_set, relation_set, triple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.triple_list = triple_list
        self.L1 = L1

        self.loss = 0

    def emb_initialize(self):  # 为 entity 和 relation 生成 [id，向量]的字典，其中向量随机生成并归一化

        relation_dict = {}
        entity_dict = {}

        for relation in self.relation:
            r_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)

        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
                                           6 / math.sqrt(self.embedding_dim),
                                           self.embedding_dim)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

        self.relation = relation_dict
        self.entity = entity_dict

    def train(self, epochs):
        nbatches = 400
        batch_size = len(self.triple_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0

            # Sbatch:list
            Sbatch = random.sample(self.triple_list, batch_size)
            Tbatch = []

            for triple in Sbatch:
                corrupted_triple = self.Corrupt(triple)
                if (triple, corrupted_triple) not in Tbatch:
                    Tbatch.append((triple, corrupted_triple))
            self.update_embeddings(Tbatch)

            end = time.time()
            # print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            # print("loss: ", self.loss)
            # loss_ls.append(self.loss)

            # 保存临时结果
            # if epoch % 20 == 0:
            #     with codecs.open("entity_temp", "w") as f_e:
            #         for e in self.entity.keys():
            #             f_e.write(e + "\t")
            #             f_e.write(str(list(self.entity[e])))
            #             f_e.write("\n")
            #     with codecs.open("relation_temp", "w") as f_r:
            #         for r in self.relation.keys():
            #             f_r.write(r + "\t")
            #             f_r.write(str(list(self.relation[r])))
            #             f_r.write("\n")

        # print("写入文件...")
        # with codecs.open("./result/entity_20dim1", "w") as f1:
        #     for e in self.entity.keys():
        #         f1.write(e + "\t")
        #         f1.write(str(list(self.entity[e])))
        #         f1.write("\n")
        #
        # with codecs.open("./result/relation_20dim1", "w") as f2:
        #     for r in self.relation.keys():
        #         f2.write(r + "\t")
        #         f2.write(str(list(self.relation[r])))
        #         f2.write("\n")
        #
        # with codecs.open("./result/loss", "w") as f3:
        #     f3.write(str(loss_ls))
        #
        # print("写入完成")

    def Corrupt(self, triple):
        corrupted_triple = copy.deepcopy(triple)
        seed = random.random()
        if seed > 0.5:
            # 替换head
            rand_head = triple[0]
            while rand_head == triple[0]:
                rand_head = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[0] = rand_head

        else:
            # 替换tail
            rand_tail = triple[1]
            while rand_tail == triple[1]:
                rand_tail = random.sample(self.entity.keys(), 1)[0]
            corrupted_triple[1] = rand_tail
        return corrupted_triple

    def update_embeddings(self, Tbatch):
        copy_entity = copy.deepcopy(self.entity)
        copy_relation = copy.deepcopy(self.relation)

        for triple, corrupted_triple in Tbatch:
            # 取copy里的vector累积更新
            h_correct_update = copy_entity[triple[0]]
            t_correct_update = copy_entity[triple[1]]
            relation_update = copy_relation[triple[2]]

            h_corrupt_update = copy_entity[corrupted_triple[0]]
            t_corrupt_update = copy_entity[corrupted_triple[1]]

            # 取原始的vector计算梯度
            h_correct = self.entity[triple[0]]
            t_correct = self.entity[triple[1]]
            relation = self.relation[triple[2]]

            h_corrupt = self.entity[corrupted_triple[0]]
            t_corrupt = self.entity[corrupted_triple[1]]

            if self.L1:
                dist_correct = distanceL1(h_correct, relation, t_correct)
                dist_corrupt = distanceL1(h_corrupt, relation, t_corrupt)
            else:
                dist_correct = distanceL2(h_correct, relation, t_correct)
                dist_corrupt = distanceL2(h_corrupt, relation, t_corrupt)

            err = self.hinge_loss(dist_correct, dist_corrupt)

            if err > 0:
                self.loss += err

                grad_pos = 2 * (h_correct + relation - t_correct)
                grad_neg = 2 * (h_corrupt + relation - t_corrupt)

                #  梯度计算
                if self.L1:
                    for i in range(len(grad_pos)):
                        if (grad_pos[i] > 0):
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if (grad_neg[i] > 0):
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1

                #  梯度下降
                # head系数为正，减梯度；tail系数为负，加梯度
                h_correct_update -= self.learning_rate * grad_pos
                t_correct_update -= (-1) * self.learning_rate * grad_pos

                # corrupt项整体为负，因此符号与correct相反
                if triple[0] == corrupted_triple[0]:  # 若替换的是尾实体，则头实体更新两次
                    h_correct_update -= (-1) * self.learning_rate * grad_neg
                    t_corrupt_update -= self.learning_rate * grad_neg

                elif triple[1] == corrupted_triple[1]:  # 若替换的是头实体，则尾实体更新两次
                    h_corrupt_update -= (-1) * self.learning_rate * grad_neg
                    t_correct_update -= self.learning_rate * grad_neg

                # relation更新两次
                relation_update -= self.learning_rate * grad_pos
                relation_update -= (-1) * self.learning_rate * grad_neg

        # batch norm 归一化
        for i in copy_entity.keys():
            copy_entity[i] /= np.linalg.norm(copy_entity[i])
        for i in copy_relation.keys():
            copy_relation[i] /= np.linalg.norm(copy_relation[i])

        # 达到批量更新的目的
        self.entity = copy_entity
        self.relation = copy_relation

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, dist_correct - dist_corrupt + self.margin)


def query_recommendation_top_k_by_user_id(user_id, recommend_num=10):
    entity_set, relation_set, triple_list = data_loader(user_id)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=20, learning_rate=0.01, margin=1, L1=True)
    transE.emb_initialize()
    transE.train(epochs=50)

    grade_predict_dict = dict()
    predict_list = []
    for movie1 in movie_set:  # 遍历所有电影
        # 不需要求出每个电影间的相似度，只需要求出当前用户看过的电影与其他电影的相似度就可以
        # 其他电影的预测评分则通过用户评分过的电影评分和相似度进行相乘加权即可
        # 这里的电影相似度矩阵如何构造呢
        # 考虑将不将相似度存起来，而是将相似度直接与用户评分相乘
        if movie1 in current_user_grade_dict.keys():  # 滤除用户看过的电影
            continue
        movie1_vec = transE.entity[movie1]
        movie1_predict = 0
        similarity_sum = 0
        for movie2 in current_user_grade_dict.keys():  # 遍历用户看过的电影
            movie2_vec = transE.entity[movie2]
            # similarity = np.sum(movie1_vec * movie2_vec) / (np.sqrt(np.sum(movie1_vec ** 2)) * np.sqrt(np.sum(movie2_vec ** 2)))
            # 采用欧式距离并将其规约到 (0,1] 之间
            similarity = 1 / (1 + np.sqrt(np.sum((movie1_vec - movie2_vec) ** 2)))
            # 未评分电影 movie1 和评分电影 movie2 相似度与 movie1 评分
            movie1_predict += similarity * current_user_grade_dict[movie2]
            similarity_sum += similarity
        # 除去相似度之和进行归一化
        movie1_predict = movie1_predict / similarity_sum
        grade_predict_dict[movie1] = movie1_predict
        recommender_num, genres = db.query_movie_recommmender_num_and_genres_by_title(movie_dict[movie1])
        predict_list.append([movie_dict[movie1], movie1_predict, recommender_num, genres])
    predict_list = sorted(predict_list, key=(lambda movie: movie[1]), reverse=True)[:recommend_num]
    # predict_df = pd.DataFrame(predict_list, columns=["title", "grade", "recommender_num", "genres"])
    # print(predict_df)
    return predict_list
    # 能否提取出相似度矩阵呢
    # 这里要提取出相似度矩阵其实是可以的，但是比较麻烦，能不能不构造相似度矩阵，直接在计算出相似度的时候进行合并呢
    # 但是这样就需要同时计算出语义相似度和协同过滤相似度
    # 计算语义相似度的时候能通过先前已经计算得到的向量，计算出用户未看过电影A与看过电影B的相似度
    # 在这里能否计算出电影A和电影B协同过滤的相似度呢
    # 当然可以啊，因为 cf 的实现就是先计算相似度存到 Neo4j 里面，那么只要取出以 A 和 B 为首尾的 similarity 关系就可以了
    # 不对，因为 cf 实现是基于用户的协同过滤，只会计算用户的相似度，怎么可能把用户的相似度矩阵和物品的相似度矩阵加权融合
    # 现在的策略就是 1.用基于物品的协同过滤，将电影间的相似度矩阵加权相加 2.将两个算出的 TopK 进行融合
