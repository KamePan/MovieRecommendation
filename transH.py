import codecs
import numpy as np
import copy
import time
import random

from neo4j import GraphDatabase
from py2neo import Graph

import database as db
import matplotlib.pyplot as plt

entity2id = {}
relation2id = {}
relation_tph = {}
relation_hpt = {}


def data_loader(user_id):
    start_data_loader = time.time()
    uri = "neo4j://localhost:7687"
    graph = Graph(uri, auth=("neo4j", "pan151312"))
    driver = GraphDatabase.driver(uri, auth=("neo4j", "pan151312"))
    session = driver.session()

    entity_set = set()
    relation_set = set()
    triple_list = []
    movie_set = set()
    movie_dict = dict()
    current_user_grade_dict = dict()
    start = time.time()
    grades = graph.run(f"""
            MATCH (User{{id:{user_id}}})-[r:RATED]-(m:Movie) RETURN m, r
        """)
    for grade_record in grades:
        movie = grade_record["m"]
        grade = grade_record["r"]
        current_user_grade_dict[str(movie.identity)] = grade["grading"]
    end = time.time()
    print("Build User Movie Grade Dict: " + str(end - start) + "s")

    start = time.time()
    all_relations = session.run(f"""
            MATCH ()-[relation]->() RETURN relation
        """)
    end = time.time()
    print("Query All Relation: " + str(end - start) + "s")

    relation_head_cnt = dict()
    relation_tail_cnt = dict()
    # 若在处理关系的时候处理实体集，由于关系中的实体是重复的，因此会被重复添加到实体集中
    # 这个循环的复杂度也就变为了 relation 数量级的处理，加上两倍 relation 数量的头尾节点 Node 的处理
    # 之前的时间过长也是因为这里（后来根据发现时间统计发现其实这里处理耗费时间不长）
    start = time.time()
    total = 0.0000
    e = time.time()
    total_interval = 0
    for relation_record in all_relations:
        s = time.time()
        total_interval += s - e
        relation = relation_record["relation"]
        start_entity = relation.start_node
        end_entity = relation.end_node
        relation_type = relation.type
        if relation_type in relation_head_cnt:
            if str(start_entity) in relation_head_cnt[relation_type]:
                relation_head_cnt[relation_type][str(start_entity.id)] += 1
            else:
                relation_head_cnt[relation_type][str(start_entity.id)] = 1
        else:
            relation_head_cnt[relation_type] = {}
            relation_head_cnt[relation_type][start_entity] = 1

        if relation_type in relation_tail_cnt:
            if str(end_entity) in relation_tail_cnt[relation_type]:
                relation_tail_cnt[relation_type][str(end_entity.id)] += 1
            else:
                relation_tail_cnt[relation_type][str(end_entity.id)] = 1
        else:
            relation_tail_cnt[relation_type] = {}
            relation_tail_cnt[relation_type][end_entity] = 1
        entity_set.add(start_entity["title"])
        entity_set.add(end_entity["title"])
        relation_set.add(relation_type)
        triple_list.append([str(start_entity.id), str(end_entity.id), relation_type])
        e = time.time()
        total += e - s
    print("Circle Interval: " + str(total_interval) + "s")
    print("循环中代码耗时: " + str(total) + "s")

    '''
    sum1 统计的是该 relation 的头结点的总数（不重复）
    sum2 统计的是该 relation 的所有头结点的总数（重复），实际上即为尾结点的总数
    sum2 / sum1 指的是结点出现在该关系的三元组中作为头结点的平均次数，即每个头结点对应的尾结点平均数量
    尾结点对应头结点平均数量也同理
    '''
    for relation_type in relation_head_cnt.keys():
        sum1, sum2 = 0, 0
        for head in relation_head_cnt[relation_type].keys():
            sum1 += 1
            sum2 += relation_head_cnt[relation_type][head]
        tph = sum2 / sum1
        relation_tph[relation_type] = tph

    for relation_type in relation_tail_cnt.keys():
        sum1, sum2 = 0, 0
        for tail in relation_tail_cnt[relation_type].keys():
            sum1 += 1
            sum2 += relation_tail_cnt[relation_type][tail]
        hpt = sum2 / sum1
        relation_hpt[relation_type] = hpt

    end = time.time()
    print("Process Relation: " + str(end - start) + "s")

    start = time.time()
    all_entities = graph.run(f"""
                MATCH (n) RETURN n
            """)
    for entity_record in all_entities:
        entity = entity_record["n"]
        entity_set.add(str(entity.identity))
    end = time.time()
    print("Process All Entity: " + str(end - start) + "s")

    start = time.time()
    all_movies = graph.run(f"""
            MATCH (m:Movie) RETURN m
        """)
    for movie_record in all_movies:
        movie = movie_record["m"]
        movie_set.add(str(movie.identity))
        movie_dict[str(movie.identity)] = movie["title"]
    end = time.time()
    print("movie_set 中电影数量: " + str(len(movie_set)))
    print("Process All Movies: " + str(end - start) + "s")

    end_data_loader = time.time()
    print("data_loader: " + str(end_data_loader - start_data_loader) + "s")
    return entity_set, relation_set, triple_list, movie_set, movie_dict, current_user_grade_dict




class TransH:
    def __init__(self, entity_set, relation_set, triple_list, embedding_dim=20, learning_rate=0.01, margin=1.0, norm=1,
                 C=1.0, epsilon=1e-5):
        self.entities = entity_set
        self.relations = relation_set
        self.triples = triple_list
        self.dimension = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.norm = norm
        self.loss = 0.0
        self.norm_relations = {}
        self.hyper_relations = {}
        self.C = C
        self.epsilon = epsilon

    def emb_initialize(self):
        entityVectorList = {}
        relationNormVectorList = {}
        relationHyperVectorList = {}
        for entity in self.entities:
            entity_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension),
                                              6.0 / np.sqrt(self.dimension),
                                              self.dimension)
            entityVectorList[entity] = entity_vector
        for relation in self.relations:
            relation_norm_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension),
                                                     6.0 / np.sqrt(self.dimension),
                                                     self.dimension)
            relation_hyper_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension),
                                                      6.0 / np.sqrt(self.dimension),
                                                      self.dimension)
            relation_norm_vector = self.normalization(relation_norm_vector)
            relation_hyper_vector = self.normalization(relation_hyper_vector)
            relationNormVectorList[relation] = relation_norm_vector
            relationHyperVectorList[relation] = relation_hyper_vector
        self.entities = entityVectorList
        self.norm_relations = relationNormVectorList
        self.hyper_relations = relationHyperVectorList

    def train(self, epochs=50, nbatches=400):
        """"""
        epoch_loss_list = dict()
        '''这里和 TransE 相同，将三元组划分为 nbatches 组，并计算出每组所含三元组数量 batch_size'''
        batch_size = int(len(self.triples) / nbatches)
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0.0
            # Normalise the embedding of the entities to 1
            for entity in self.entities:
                self.entities[entity] = self.normalization(self.entities[entity])

            '''TransE 这里不需要遍历每个组，为什么 TransH 则需要，因此先去掉该重循环'''
            # for batch in range(nbatches):
            Sbatch = random.sample(self.triples, batch_size)
            Tbatch = []
            '''这里对 Sbatch 中的三元组都进行破坏，即构造负采样中的错误三元组'''
            '''将错误三元组加入 Tbatch 中，得到错误三元组集合 Tbatch'''
            for triple in Sbatch:
                corrupted_triple = self.corrupt(triple)
                if (triple, corrupted_triple) not in Tbatch:
                    Tbatch.append((triple, corrupted_triple))
            '''通过梯度下降算法更新梯度与参数'''
            self.update_triple_embedding(Tbatch)
            '''统计损失函数值随 epoch 的变化情况，用于绘制变化情况表'''
            epoch_loss_list[epoch] = self.loss

            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("running loss: ", self.loss)
        '''显示损失函数 loss 随 epoch 变化表'''
        plt.plot(epoch_loss_list.keys(), epoch_loss_list.values())
        plt.show()

    def corrupt(self, triple):
        corrupted_triple = copy.deepcopy(triple)
        pr = np.random.random(1)[0]
        p = relation_tph[corrupted_triple[2]] / \
            (relation_tph[corrupted_triple[2]] + relation_hpt[corrupted_triple[2]])
        '''
        这里关于p的说明 tph 表示每一个头结对应的平均尾节点数 hpt 表示每一个尾节点对应的平均头结点数
        当tph > hpt 时 更倾向于替换头 反之则跟倾向于替换尾实体

        举例说明 
        在一个知识图谱中，一共有10个实体 和n个关系，如果其中一个关系使两个头实体对应五个尾实体，
        那么这些头实体的平均 tph为2.5，而这些尾实体的平均 hpt只有0.4，
        则此时我们更倾向于替换头实体，
        因为替换头实体才会有更高概率获得正假三元组，如果替换头实体，获得正假三元组的概率为 8/9 而替换尾实体获得正假三元组的概率只有 5/9
        '''
        if pr < p:
            # change the head entity
            corrupted_triple[0] = random.sample(self.entities.keys(), 1)[0]
            while corrupted_triple[0] == triple[0]:
                corrupted_triple[0] = random.sample(self.entities.keys(), 1)[0]
        else:
            # change the tail entity
            corrupted_triple[1] = random.sample(self.entities.keys(), 1)[0]
            while corrupted_triple[1] == triple[1]:
                corrupted_triple[1] = random.sample(self.entities.keys(), 1)[0]
        return corrupted_triple

    def normalization(self, vector):
        return vector / np.linalg.norm(vector)

    def norm_l2(self, h, r_norm, r_hyper, t):
        return np.sum(np.square(h - np.dot(r_norm, h) * r_norm + r_hyper - t + np.dot(r_norm, t) * r_norm))

    def norm_l1(self, h, r, t):
        return np.sum(np.fabs(h + r - t))

    # 知乎上询问过清华的大佬对于软约束项的建议 模长约束对结果收敛有影响，但是正交约束影响很小所以模长约束保留，正交约束可以不加
    def scale_entity(self, h, t, h_c, t_c):
        return np.linalg.norm(h) ** 2 - 1 + np.linalg.norm(t) ** 2 - 1 + np.linalg.norm(h_c) ** 2 - 1 + np.linalg.norm(
            t_c) ** 2 - 1

    def orthogonality(self, norm, hyper):
        return np.dot(norm, hyper) ** 2 / np.linalg.norm(hyper) ** 2 - self.epsilon ** 2

    def update_triple_embedding(self, Tbatch):
        # deepcopy 可以保证，即使list嵌套list也能让各层的地址不同， 即这里copy_entity 和
        # entitles中所有的elements都不同
        copy_entity = copy.deepcopy(self.entities)
        copy_norm_relation = copy.deepcopy(self.norm_relations)
        copy_hyper_relation = copy.deepcopy(self.hyper_relations)

        for correct_sample, corrupted_sample in Tbatch:

            correct_copy_head = copy_entity[correct_sample[0]]
            correct_copy_tail = copy_entity[correct_sample[1]]
            relation_norm_copy = copy_norm_relation[correct_sample[2]]
            relation_hyper_copy = copy_hyper_relation[correct_sample[2]]

            corrupted_copy_head = copy_entity[corrupted_sample[0]]
            corrupted_copy_tail = copy_entity[corrupted_sample[1]]

            correct_head = self.entities[correct_sample[0]]
            correct_tail = self.entities[correct_sample[1]]
            relation_norm = self.norm_relations[correct_sample[2]]
            relation_hyper = self.hyper_relations[correct_sample[2]]

            corrupted_head = self.entities[corrupted_sample[0]]
            corrupted_tail = self.entities[corrupted_sample[1]]

            # calculate the distance of the triples
            correct_distance = self.norm_l2(correct_head, relation_norm, relation_hyper, correct_tail)
            corrupted_distance = self.norm_l2(corrupted_head, relation_norm, relation_hyper, corrupted_tail)

            loss = self.margin + correct_distance - corrupted_distance
            loss1 = self.scale_entity(correct_head, correct_tail, corrupted_head, corrupted_tail)
            # loss2 = self.orthogonality(relation_norm, relation_hyper)

            if loss > 0:
                self.loss += loss
                i = np.ones(self.dimension)
                correct_gradient = 2 * (correct_head - np.dot(relation_norm, correct_head) * relation_norm + relation_hyper - correct_tail +
                                        np.dot(relation_norm, correct_tail) * relation_norm) * (i - relation_norm ** 2)
                corrupted_gradient = 2 * (corrupted_head - np.dot(relation_norm, corrupted_head) *
                                          relation_norm + relation_hyper - corrupted_tail +
                                          np.dot(relation_norm, corrupted_tail) *
                                          relation_norm) * (i - relation_norm ** 2)
                hyper_gradient = 2 * (correct_head - np.dot(relation_norm, correct_head) *
                                      relation_norm + - correct_tail +
                                      np.dot(relation_norm, correct_tail)
                                      * relation_norm) - 2 * (
                                             corrupted_head - np.dot(relation_norm, corrupted_head)
                                             * relation_norm + - corrupted_tail +
                                             np.dot(relation_norm, corrupted_tail) *
                                             relation_norm)
                norm_gradient = 2 * (correct_head - np.dot(relation_norm, correct_head) *
                                     relation_norm +relation_hyper - correct_tail +
                                     np.dot(relation_norm, correct_tail) *
                                     relation_norm) * (correct_tail - correct_head) * 2 * relation_norm - 2 * (
                                            corrupted_head - np.dot(relation_norm, corrupted_head) * relation_norm +
                                            relation_hyper - corrupted_tail +
                                            np.dot(relation_norm, corrupted_tail) *
                                            relation_norm) * (corrupted_tail - corrupted_head) * 2 * relation_norm

                correct_copy_head -= self.learning_rate * correct_gradient
                relation_norm_copy -= self.learning_rate * norm_gradient
                relation_hyper_copy -= self.learning_rate * hyper_gradient
                correct_copy_tail -= -1 * self.learning_rate * correct_gradient

                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replaces the tail entity, the head entity's embedding need to be updated twice
                    correct_copy_head -= -1 * self.learning_rate * corrupted_gradient
                    corrupted_copy_tail -= self.learning_rate * corrupted_gradient
                elif correct_sample[1] == corrupted_sample[1]:
                    # if corrupted_triples replaces the head entity, the tail entity's embedding need to be updated twice
                    corrupted_copy_head -= -1 * self.learning_rate * corrupted_gradient
                    correct_copy_tail -= self.learning_rate * corrupted_gradient

                # normalising these new embedding vector, instead of normalising all the embedding together
                copy_entity[correct_sample[0]] = self.normalization(correct_copy_head)
                copy_entity[correct_sample[1]] = self.normalization(correct_copy_tail)
                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replace the tail entity, update the tail entity's embedding
                    copy_entity[corrupted_sample[1]] = self.normalization(corrupted_copy_tail)
                elif correct_sample[1] == corrupted_sample[1]:
                    # if corrupted_triples replace the head entity, update the head entity's embedding
                    copy_entity[corrupted_sample[0]] = self.normalization(corrupted_copy_head)
                # the paper mention that the relation's embedding don't need to be normalised
                copy_norm_relation[correct_sample[2]] = self.normalization(relation_norm_copy)
                copy_hyper_relation[correct_sample[2]] = relation_hyper_copy
                # copy_relation[correct_sample[2]] = self.normalization(relation_copy)

                # self.loss += loss + self.C * loss1
                # if loss1 > 0:
                #     if np.linalg.norm(correct_head) > 1:
                #         hcg =  2 * correct_head
                #     if np.linalg.norm(correct_tail)> 1:
                #         tcg =  2 * correct_tail
                #     if np.linalg.norm(corrupted_head)> 1:
                #         hcorg =  2 * corrupted_head
                #     if np.linalg.norm(corrupted_tail) > 1:
                #         tcorg =  2 * corrupted_tail
                #
                # correct_copy_head -= self.learning_rate * correct_gradient
                # relation_norm_copy -= self.learning_rate * norm_gradient
                # relation_hyper_copy -=  self.learning_rate * hyper_gradient
                # correct_copy_tail -= -1 * self.learning_rate * correct_gradient
                #
                # if correct_sample[0] == corrupted_sample[0]:
                #     # if corrupted_triples replaces the tail entity, the head entity's embedding need to be updated twice
                #     correct_copy_head -= -1 * self.learning_rate * corrupted_gradient
                #     corrupted_copy_tail -= self.learning_rate * corrupted_gradient
                # elif correct_sample[1] == corrupted_sample[1]:
                #     # if corrupted_triples replaces the head entity, the tail entity's embedding need to be updated twice
                #     corrupted_copy_head -= -1 * self.learning_rate * corrupted_gradient
                #     correct_copy_tail -= self.learning_rate * corrupted_gradient
                #
                # correct_copy_head -= self.learning_rate * hcg
                # correct_copy_tail -=  self.learning_rate * tcg
                #
                # copy_entity[correct_sample[0]] = correct_copy_head
                # copy_entity[correct_sample[1]] = correct_copy_tail
                # if correct_sample[0] == corrupted_sample[0]:
                #     # if corrupted_triples replace the tail entity, update the tail entity's embedding
                #     copy_entity[corrupted_sample[1]] = corrupted_copy_tail
                # elif correct_sample[1] == corrupted_sample[1]:
                #     # if corrupted_triples replace the head entity, update the head entity's embedding
                #     copy_entity[corrupted_sample[0]] = corrupted_copy_head
                # copy_norm_relation[correct_sample[2]] = relation_norm_copy
                # copy_hyper_relation[correct_sample[2]] = relation_hyper_copy

        self.entities = copy_entity
        self.norm_relations = copy_norm_relation
        self.hyper_relations = copy_hyper_relation


def query_recommendation_top_k_by_user_id(user_id, recommend_num=10):
    start = time.time()
    entity_set, relation_set, triple_list, movie_set, movie_dict, current_user_grade_dict = data_loader(user_id)
    print("load file...")
    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity_set), len(relation_set), len(triple_list)))

    start_translate_train = time.time()
    transH = TransH(entity_set, relation_set, triple_list, embedding_dim=20, learning_rate=0.01, margin=1)
    transH.emb_initialize()
    transH.train(epochs=20)
    end_translate_train = time.time()
    print("TransE train time: " + str(end_translate_train - start_translate_train) + "s")

    start_predict = time.time()
    predict_list = []
    for movie1 in movie_set:  # 遍历所有电影
        # 不需要求出每个电影间的相似度，只需要求出当前用户看过的电影与其他电影的相似度就可以
        # 其他电影的预测评分则通过用户评分过的电影评分和相似度进行相乘加权即可
        # 这里的电影相似度矩阵如何构造呢
        # 考虑将不将相似度存起来，而是将相似度直接与用户评分相乘
        if movie1 in current_user_grade_dict.keys():  # 滤除用户看过的电影
            continue
        movie1_vec = transH.entities[movie1]
        movie1_predict = 0
        similarity_sum = 0
        for movie2 in current_user_grade_dict.keys():  # 遍历用户看过的电影
            movie2_vec = transH.entities[movie2]
            # similarity = np.sum(movie1_vec * movie2_vec) / (np.sqrt(np.sum(movie1_vec ** 2)) * np.sqrt(np.sum(movie2_vec ** 2)))
            # 采用欧式距离并将其规约到 (0,1] 之间
            similarity = 1 / (1 + np.sqrt(np.sum((movie1_vec - movie2_vec) ** 2)))
            # 未评分电影 movie1 和评分电影 movie2 相似度与 movie1 评分
            movie1_predict += similarity * current_user_grade_dict[movie2]
            similarity_sum += similarity
        # 除去相似度之和进行归一化
        movie1_predict = movie1_predict / similarity_sum
        predict_list.append([movie_dict[movie1], movie1_predict])
    predict_list = sorted(predict_list, key=(lambda movie: movie[1]), reverse=True)[:recommend_num]
    # 放在这里做查询就能够只做 recommend_num 次数的查询
    for predict_movie in predict_list:
        predict_movie_title = predict_movie[0]
        recommender_num, genres = db.query_movie_recommmender_num_and_genres_by_title(predict_movie_title)
        predict_movie.append(recommender_num)
        predict_movie.append(genres)
    end_predict = time.time()
    print("build transH predict list: " + str(end_predict - start_predict) + "s")
    end = time.time()
    print("query_recommendation_top_k_by_user_id: " + str(end - start) + "s")
    return predict_list


if __name__ == '__main__':
    entity_set, relation_set, triple_list, movie_set, movie_dict, current_user_grade_dict = data_loader(440)
    transH = TransH(entity_set, relation_set, triple_list, embedding_dim=50, learning_rate=0.01, margin=1.0, norm=1)
    transH.emb_initialize()
    transH.train(epochs=200)
