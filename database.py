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
    result = session.run(f"""MATCH (g:Genre) RETURN g.genre AS genre""")
    return result


def query_rated_movie_by_user_id(user_id):
    result = session.run(f"""
            MATCH (u1:User{{id:{user_id}}})-[r:RATED]-(m:Movie)
            RETURN m.title AS title, r.grading as grade
            ORDER BY grade DESC
        """)
    return result


def delete_similarity():
    session.run(f"""
            MATCH (u1: User)-[s:SIMILARITY]-(u2:User)
            DELETE s
        """)


def build_similarity(user_id, movies_common, threshold_sim):
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


def calculate_collaborative_filter_recommendation_rank(user_id, k, users_common, m):
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
    return result


def query_movie_recommmender_num_and_genres_by_title(title):
    query_result = graph.run(f"""
            MATCH (u:User)-[:RATED]-(m:Movie{{title:\"{title}\"}})
            OPTIONAL MATCH (m)--(g:Genre)
            WITH COUNT(u) AS recommender_num, COLLECT(DISTINCT g.genre) AS genres 
            RETURN recommender_num, genres
        """).data()[0]
    recommender_num = query_result["recommender_num"]
    genres = query_result["genres"]
    return recommender_num, genres


if __name__ == '__main__':
    query_movie_recommmender_num_and_genres_by_title("Dogma")