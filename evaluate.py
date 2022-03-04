import pandas


def evaluate():
    """
    将 ratings 数据集分为 train 训练集及 test 测试集两个集合，比例一般为 3：7
    获取物品的预测评分和物品的预测评分
    $$ 如何检测获得均方根误差 RMSE
    预测用户的评分时，
    对于每个用户的评分，假设对 n 个电影有评分，则将 ceil(0.7 * n) 作为用户评分训练集，将 floor(0.3 * n) 作为测试集
    ***** 物品的语义相似度提前计算存起来，不然到时候训练时间过长 *****
    将语义相似度存放在 ([movie1, movie2], similarity) 的字典数据结构中
    由于是要测试电影评分，我们要预测的电影只需要是测试集中电影的评分就可以
    具体到我们的算法中，
    由于要预测算法的效率，因此需要统计对所有用户评分的效果。
    ***** 如果要提升训练的效率，就需要提前算出每个电影与其他电影的协同过滤相似度 *****
    MATCH Movie - RATED - User - RATED - Movie
    通过评分加权得到 Movie 的协同过滤物品相似度，与语义相同将这个相似度存放在 ([movie1, movie2], similarity) 的字典数据结构中

    for user in users: # 对于每个用户
        获得 user 的评分向量，按照比例将评分按照 3:7 分为测试集和训练集
        for test_movie in test_set:
            for other_movie in movie_set:
                从字典中得到 (test_movie, other_movie) 的协同过滤相似度和语义相似度
                将协同过滤物品相似度与语义相似度加权融合得到融合相似度
            for movie_train in train_set:
                predict_rating += Similarity_train_test * Rating_Movie_train
            并进行归一化
        计算均方根误差：RMSE = SUM(SQRT((movie_test_rating - predict_rating)^2))
    """
    grade_dataframe = pandas.read_csv('./dataset/out_grade_new.csv')
    user_set = set()
    movie_set = set()
    user_movie_rating_dict = dict()
    for index, row in grade_dataframe.iterrows():
        user_id = "\"" + row['user_id'] + "\""
        movie_title = row['title']
        grade = row['grade']
        user_set.add(user_id)
        movie_set.add(movie_title)
        if movie_title not in user_movie_rating_dict[user_id].keys():
            user_movie_rating_dict[user_id] = dict()
        user_movie_rating_dict[user_id][movie_title] = grade
