from neo4j import GraphDatabase
import pandas as pd
import database as db

k = 10
movies_common = 3
users_common = 3
threshold_sim = 0.9


def query_recommendation_top_k_by_user_id(user_id, recommend_num=10):
    db.delete_user_similarity()
    db.build_user_similarity(user_id, movies_common, threshold_sim)
    q = db.calculate_collaborative_filter_recommendation_rank(user_id, k, users_common, recommend_num)
    db.delete_user_similarity()

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


def query_current_user_rated_movie_similarity_table(user_id):
    db.delete_item_similarity()
    result = db.build_item_similarity(user_id=user_id, users_common=users_common, threshold_sim=threshold_sim)
    db.delete_item_similarity()
    return result
