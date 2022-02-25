from neo4j import GraphDatabase
import pandas as pd
import database as db

k = 10
movies_common = 3
users_common = 3
threshold_sim = 0.9


def query_recommendation_top_k_by_user_id(user_id, recommend_num=10):
    # q = db.query_rated_movie_by_user_id(user_id)
    # print(f"Ratings of {user_id} are following...")
    # result = []
    # for r in q:
    #     result.append([r["title"], r["grade"]])
    # if len(result) == 0:
    #     print("no ratings found")
    # else:
    #     df = pd.DataFrame(result, columns=["title", "grade"])
    #     print(df.to_string(index=False))
    # print()

    db.delete_similarity()
    db.build_similarity(user_id, movies_common, threshold_sim)

    q = db.calculate_collaborative_filter_recommendation_rank(user_id, k, users_common, recommend_num)
    db.delete_similarity()
    print("Recommended Movies:\n")
    result = []
    for r in q:
        result.append([r["title"], r["grade"], r["num"], r["genres"]])
    if len(result) == 0:
        print("No recommendations found")
        return
    df = pd.DataFrame(result, columns=["title", "avg_grade", "num_recommenders", "genres"])
    print(df.to_string(index=False))
    return result

