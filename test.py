from neo4j import GraphDatabase
import pandas as pd
import sklearn.model_selection
import math
import matplotlib.pyplot as plt

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


def test_tuple_dict():
    tuple_dict = dict()
    tuple_dict[("1", "2")] = 3
    test_tuple = ("2", "1")
    print(tuple_dict[test_tuple])


def test_cf():
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "pan151312"), max_connection_lifetime=200)
    session = driver.session()
    query_result = session.run(f"""
                    MATCH (m1:Movie{{title:"故事的故事"}})-[r1:RATED]-(u:User)-[r2:RATED]-(m2:Movie{{title:"人类之子"}})
                    WITH 
                    SUM(r1.grading * r2.grading) / (SQRT(SUM(r1.grading^2)) * SQRT(SUM(r2.grading^2))) AS similarity,
                    COUNT(u) AS user_cnt
                    WHERE user_cnt > 0
                    RETURN similarity
                """)

    print(len(query_result.data()))


def test_train_test_split():
    data_set = set()
    data_set.add("1")
    data_set.add("4")
    data_set.add("9")
    data_set.add("10")
    data_set.add("23")
    data_list = ["2", "4", "6", "12", "53"]
    data_dict = {"a":"SD", "b":"sdf"}
    train_set, test_set = sklearn.model_selection.train_test_split(list(data_dict.keys()), test_size=0.3)
    print(train_set)
    print(test_set)


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


def test_RMSE():
    records1 = [3, 4, 5, 3, 12]
    records2 = [2, 4, 6, 1, 4]
    rmse = get_rmse(records1, records2)  # 0.81
    print(rmse)


def test_build_RMSE_Graph():
    transE_RMSE = [1.0519, 1.0519, 1.0521, 1.0524, 1.0529, 1.0535]
    transH_RMSE = [1.050689, 1.0507, 1.0509, 1.0513, 1.0519, 1.0526]
    alpha_arr = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.plot(alpha_arr, transE_RMSE)
    plt.plot(alpha_arr, transH_RMSE)
    # RMSE = [0, 0, 0, 0, 0, 0]
    # for index in range(0, 6):
    #     origin_list = origin_list_arr[index]
    #     predict_list = predict_list_arr[index]
    #     RMSE[index] = get_rmse(origin_list, predict_list)
    # plt.plot(alpha_arr, RMSE)
    plt.ylim((1.0500, 1.0550))
    plt.xlabel('alpha')  # x轴标题
    plt.ylabel('RMSE')  # y轴标题
    # plt.legend(['TransE+Item-CF'])
    plt.legend(['TransE+Item-CF', 'TransH+Item-CF '])
    # plt.title('Use Euclidean Metric To Measure CF Item Similarity')
    plt.title('Trans RMSE Graph')
    plt.show()


if __name__ == '__main__':
    test_build_RMSE_Graph()
