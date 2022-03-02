import time

from neo4j import GraphDatabase
from py2neo import Graph
import pandas as pd

uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "pan151312"))
session = driver.session()

graph = Graph(uri, auth=("neo4j", "pan151312"))


def load_data():
    session.run("""MATCH ()-[r]->() DELETE r""")
    session.run("""MATCH (r) DELETE r""")

    print("Loading movies...")
    session.run("""
        LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_movies.csv" AS csv
        CREATE (:Movie {title: csv.title})        
    """)

    print("Loading gradings...")
    session.run("""
        LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_grade.csv" AS csv
        MERGE (m:Movie {title: csv.title})
        MERGE (u:User {id: toInteger(csv.user_id)})
        CREATE (u)-[:RATED {grading: toInteger(csv.grade)}]->(m)        
    """)

    print("Loading genres...")
    session.run("""
        LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_genre.csv" AS csv
        MERGE (m:Movie {title: csv.title})
        MERGE (g:Genre {genre: csv.genre})
        CREATE (m)-[:HAS_GENRE]->(g)        
    """)

    print("Loading keywords...")
    session.run("""
        LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_keyword.csv" AS csv
        MERGE (m:Movie {title: csv.title})
        MERGE (k:Keyword {keyword: csv.keyword})
        CREATE (m)-[:HAS_KEYWORD]->(k)        
    """)

    print("Loading productors...")
    session.run("""
        LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_productor.csv" AS csv
        MERGE (m:Movie {title: csv.title})
        MERGE (p:Productor {name: csv.productor})
        CREATE (m)-[:PRODUCED_BY]->(p)        
        """)


def query_all_genres():
    start = time.time()
    result = session.run(f"""MATCH (g:Genre) RETURN g.genre AS genre""")
    end = time.time()
    print("query_all_genres: " + str(end - start) + "s")
    return result


def query_rated_movie_by_user_id(user_id):
    start = time.time()
    result = session.run(f"""
            MATCH (u1:User{{id:{user_id}}})-[r:RATED]-(m:Movie)
            RETURN m.title AS title, r.grading as grade
            ORDER BY grade DESC
        """)
    end = time.time()
    print("query_rated_movie_by_user_id: " + str(end - start) + "s")
    return result


def delete_user_similarity():
    session.run(f"""
            MATCH (u1: User)-[s:SIMILARITY]-(u2:User)
            DELETE s
        """)


def build_user_similarity(user_id, movies_common, threshold_sim):
    start = time.time()
    # 用户余弦相似度计算
    session.run(f"""
            MATCH (u1:User{{id:{user_id}}})-[r1:RATED]-(m:Movie)-[r2:RATED]-(u2:User)
            WITH
                u1, u2,
                COUNT(m) AS movies_common,
                SUM(r1.grading * r2.grading) / (SQRT(SUM(r1.grading^2)) * SQRT(SUM(r2.grading^2))) AS sim
            WHERE movies_common >= {movies_common} AND sim > {threshold_sim}
            MERGE (u1)-[s:SIMILARITY]-(u2)
            SET s.sim = sim
        """)
    end = time.time()
    print("build_similarity: " + str(end - start) + "s")


# 这里对某个 user 进行推荐似乎没法计算物品相似度，如果要计算物品相似度，那么就需要
# movie rate user rate movie 这样的关系，然后通过两个 rate 评分计算物品相似度。
# 基于物品的协同过滤适用于物品少用户多的情况，比如我们这里的电影推荐系统。
def build_item_similarity(user_id, users_common, threshold_sim):
    start = time.time()
    current_user_grade_dict = dict()
    query_rated_movie_by_user_id(user_id)
    grades = graph.run(f"""
            MATCH (User{{id:{user_id}}})-[r:RATED]-(m:Movie) RETURN m, r
        """)
    for grade_record in grades:
        movie = grade_record["m"]
        grade = grade_record["r"]
        current_user_grade_dict[str(movie.identity)] = grade["grading"]

    movie_similarity_table = dict()
    # users_common 指的是同时看过两个电影的 user 的数量
    for movie_id in current_user_grade_dict.keys():
        similarities = session.run(f"""
                    MATCH (m1:Movie)-[r1:RATED]-(u:User)-[r2:RATED]-(m2:Movie)
                    WHERE id(m1) = {int(movie_id)}
                    WITH
                        m1, m2, 
                        COUNT(u) AS users_common,
                        SUM(r1.grading * r2.grading) / (SQRT(SUM(r1.grading^2)) * SQRT(SUM(r2.grading^2))) AS sim
                    WHERE users_common >= {users_common}
                    MERGE (m1)-[s:SIMILARITY]-(m2)
                    SET s.sim = sim
                    RETURN m1, m2, s
                """)
        for sim_record in similarities:
            movie1_id = str(sim_record["m1"].id)
            movie2_id = str(sim_record["m2"].id)
            similarity = sim_record["s"]
            if movie1_id not in movie_similarity_table.keys():
                movie_similarity_table[movie1_id] = dict()
            movie_similarity_table[movie1_id][movie2_id] = similarity["sim"]
    end = time.time()
    print("build_item_similarity: " + str(end - start) + "s")
    return movie_similarity_table


def delete_item_similarity():
    session.run(f"""
            MATCH (m1:Movie)-[s:SIMILARITY]-(m2:Movie)
            DELETE s
        """)


# 如果之前通过评分矩阵计算的是物品相似度，则可以融合物品相似度与语义相似度矩阵进行评估
def calculate_collaborative_filter_recommendation_rank(user_id, k, users_common, m):
    start = time.time()
    result = session.run(f"""
            MATCH (u1:User{{id:{user_id}}})-[s:SIMILARITY]-(u2:User) 
            WITH u1, u2, s 
            ORDER BY s.sim DESC LIMIT {k} 
            MATCH (m:Movie)-[r:RATED]-(u2) 
            OPTIONAL MATCH (g:Genre)--(m) 
            WITH 
                m.title AS title, 
                SUM(r.grading * s.sim) / SUM(s.sim) AS grade, 
                COUNT(u2) AS num, 
                COLLECT(DISTINCT g.genre) AS genres 
            WHERE num >= {users_common} 
            RETURN title, grade, num, genres 
            ORDER BY grade DESC, num DESC 
            LIMIT {m}
        """)
    end = time.time()
    print("calculate_collaborative_filter_recommendation_rank: " + str(end - start) + "s")
    return result


def query_movie_recommmender_num_and_genres_by_title(title):
    # start = time.time()
    # query_result = graph.run(f"""
    #         MATCH (u:User)-[:RATED]-(m:Movie{{title:\"{title}\"}})
    #         OPTIONAL MATCH (m)--(g:Genre)
    #         WITH COUNT(u) AS recommender_num, COLLECT(DISTINCT g.genre) AS genres
    #         RETURN recommender_num, genres
    #     """).data()[0]
    query_result = graph.run(f"""
                MATCH (m:Movie{{title:\"{title}\"}})--(g:Genre)
                RETURN COLLECT(DISTINCT g.genre) AS genres 
            """).data()[0]
    # recommender_num = query_result["recommender_num"]
    genres = query_result["genres"]
    # end = time.time()
    # print("query_movie_recommmender_num_and_genres_by_title: " + str(end - start) + "s")
    return 0, genres


if __name__ == '__main__':
    load_data()
    # delete_item_similarity()
    # build_item_similarity(440, 2, 0.9)
    # delete_item_similarity()
    # query_movie_recommmender_num_and_genres_by_title("Dogma")