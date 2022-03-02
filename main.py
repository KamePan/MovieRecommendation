import cf
import transE
import pandas as pd
import math
import time
import transH


def recommend_merge_top_k(user_id, recommend_num, trans=0):
    start = time.time()
    cf_top_k = cf.query_recommendation_top_k_by_user_id(user_id=user_id, recommend_num=recommend_num)
    trans_top_k = list()
    if trans == 0:
        trans_top_k = transE.query_recommendation_top_k_by_user_id(user_id=user_id, recommend_num=recommend_num)
    elif trans == 1:
        trans_top_k = transH.query_recommendation_top_k_by_user_id(user_id=user_id, recommend_num=recommend_num)
    print("协同过滤推荐 TopK: ")
    print(pd.DataFrame(cf_top_k, columns=["title", "grade", "recommend_num", "genres"]).to_string(index=False))
    print("语义推荐 TopK: ")
    print(pd.DataFrame(trans_top_k, columns=["title", "grade", "recommend_num", "genres"]).to_string(index=False))
    p = 5
    q = 5
    n = math.ceil(recommend_num * (p / (p + q)))
    migrated_top_k = cf_top_k
    for i in range(n):
        migrated_top_k[recommend_num - n + i] = trans_top_k[i]
    print("融合语义和协同过滤的推荐 TopK: ")
    print(pd.DataFrame(migrated_top_k, columns=["title", "grade", "recommender_num", "genres"]).to_string(index=False))
    end = time.time()
    print("算法执行时间：" + str(end - start) + "s")
    return migrated_top_k


def recommend_merge_similarity(user_id, recommend_num):
    start = time.time()
    migrated_top_k = transE.query_merged_recommendation_top_k_by_user_id(user_id=user_id, recommend_num=recommend_num)
    print("融合语义和协同过滤的推荐 TopK: ")
    print(pd.DataFrame(migrated_top_k, columns=["title", "grade", "recommender_num", "genres"]).to_string(index=False))
    end = time.time()
    print("算法执行时间：" + str(end - start) + "s")
    return migrated_top_k


if __name__ == '__main__':
    recommend_merge_top_k(user_id=440, recommend_num=10)
    # recommend_merge_similarity(user_id=440, recommend_num=10)
