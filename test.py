from neo4j import GraphDatabase
import pandas as pd

uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "pan151312"))


def test1(user_id=201):
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

def recommend(user_id=201):
    with driver.session() as session:
        q = session.run(f"""
                        MATCH (u1:User{{id:{user_id}}})-[s:SIMILARITY]-(u2:User)
                        WITH u1, u2, s
                        ORDER BY s.sim DESC LIMIT 10
                        MATCH (m:Movie)-[r:RATED]-(u2)
                        OPTIONAL MATCH (g:Genre)--(m)
                        WITH u1, u2, s, m, r, COLLECT(DISTINCT g.genre) AS gen
                        WITH
                            m.title AS title,
                            SUM(r.grading * s.sim) / SUM(s.sim) AS grade,
                            COUNT(u2) AS num,
                            gen
                        WHERE num >= 2
                        RETURN title, grade, num, gen
                        ORDER BY grade DESC, num DESC
                        LIMIT 10
                    """)
        print("Recommended Movies:")
        result = []
        for r in q:
            result.append([r["title"], r["grade"], r["num"], r["gen"]])
        if len(result) == 0:
            print("No recommendations found")
        else:
            df = pd.DataFrame(result, columns=["title", "avg_grade", "num recommenders", "genres"])
            dfStyler = df.style.set_properties(**{'text-align': 'left'})
            dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
            print(df.to_string(index=False))


def deleteSimilarity():
    with driver.session() as session:
        session.run(f"""
            MATCH (u1: User)-[s:SIMILARITY]-(u2:User)
            DELETE s
        """)


def buildSimilarity(user_id=201):
    with driver.session() as session:
        # 用户余弦相似度计算
        session.run(f"""
            MATCH (u1:User{{id:{user_id}}})-[r1:RATED]-(m:Movie)-[r2:RATED]-(u2:User)
            WITH
                u1, u2,
                COUNT(m) AS movies_common,
                SUM(r1.grading * r2.grading) / (SQRT(SUM(r1.grading^2)) * SQRT(SUM(r2.grading^2))) AS sim
            WHERE movies_common >= 3 AND sim > 0.9
            MERGE (u1)-[s:SIMILARITY]-(u2)
            SET s.sim = sim
        """)
    print("build complete.")

def compare(elem):
    return elem[0][0]


def testListSort():
    list_a = [["a", 1], ["c", -1], ["b", 3]]
    print(list_a)
    print(sorted(list_a, key=(lambda x: x[1])))


if __name__ == '__main__':
    # deleteSimilarity()
    # buildSimilarity(440)
    # recommend(440)
    testListSort()
