from neo4j import GraphDatabase
import pandas as pd

uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "pan151312"))

k = 10
movies_common = 3
users_common = 2
threshold_sim = 0.9


def load_data():
    with driver.session() as session:
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


def queries():
    while True:
        user_id = int(input("请输入要为哪位用户推荐电影，输入其 ID 即可\n"))
        m = int(input("为用户推荐多少个电影呢？\n"))

        genres = []
        if int(input("是否过滤掉不喜欢类型（输入0或1）\n")):
            with driver.session() as session:
                try:
                    q = session.run(f"""MATCH (g:Genre) RETURN g.genre AS genre""")
                    result = []
                    for i, r in enumerate(q):
                        result.append(r["genre"])
                    df = pd.DataFrame(result, columns=["genre"])
                    print(df)
                    inp = input("输入不喜欢类型索引\n")
                    if len(inp) != 0:
                        inp = inp.split(" ")
                        genres = [df["genre"].iloc[int(x)] for x in inp]
                except:
                    print("Error")

        with driver.session() as session:
            q = session.run(f"""
                MATCH (u1:User{{id:{user_id}}})-[r:RATED]-(m:Movie)
                RETURN m.title AS title, r.grading as grade
                ORDER BY grade DESC
            """)

            print(f"Ratings of {user_id} are following...")
            result = []
            for r in q:
                result.append([r["title"], r["grade"]])
            if len(result) == 0:
                print("no ratings found")
            else:
                df = pd.DataFrame(result, columns=["title", "grade"])
                print(df.to_string(index=False))
            print()

            session.run(f"""
                MATCH (u1: User)-[s:SIMILARITY]-(u2:User)
                DELETE s
            """)

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

            Q_GENRE = ""
            if len(genres) > 0:
                Q_GENRE = "AND ((SIZE(gen) > 0) AND "
                Q_GENRE += "(ANY(x IN " + str(genres) + " WHERE x IN gen))"
                Q_GENRE += ")"

            q = session.run(f"""
                MATCH (u1:User{{id:{user_id}}})-[s:SIMILARITY]-(u2:User) 
                WITH u1, u2, s 
                ORDER BY s.sim DESC LIMIT {k} 
                MATCH (m:Movie)-[r:RATED]-(u2) 
                OPTIONAL MATCH (g:Genre)--(m) 
                WITH u1, u2, s, m, r, COLLECT(DISTINCT g.genre) AS gen 
                WHERE NOT ((m)-[:RATED]-(u1)) {Q_GENRE} 
                WITH 
                    m.title AS title, 
                    SUM(r.grading * s.sim) / SUM(s.sim) AS grade, 
                    COUNT(u2) AS num, 
                    gen 
                WHERE num >= {users_common} 
                RETURN title, grade, num, gen 
                ORDER BY grade DESC, num DESC 
                LIMIT {m}
            """)

            print("Recommended Movies:\n")
            result = []
            for r in q:
                result.append([r["title"], r["grade"], r["num"], r["gen"]])
            if len(result) == 0:
                print("No recommendations found")
                continue
            df = pd.DataFrame(result, columns=["title", "avg grade", "num recommenders", "genres"])
            print(df.to_string(index=False))


if __name__ == '__main__':
    #load_data()
    queries()
