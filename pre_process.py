import pandas as pd
import json
import re

intersection_movie_set = set()


def Netflix(MAX_USER = 1000):
    d_movie = dict()
    s_movie = set()
    out_movies = open("./dataset/out_movies.csv", "w")
    out_movies.write("title\n")

    # 这里用出现在 movie_titles（Netflix Prize）里面的数据建立字典和集合
    for line in open("./dataset/movie_titles.csv", "r", encoding='ISO-8859-1'):
        line = line.strip().split(',')
        movie_id = int(line[0])
        original_title = line[2].replace("\"", "")
        title = "\"" + original_title + "\""
        print(title)
        d_movie[movie_id] = title
        if title in s_movie or original_title not in intersection_movie_set:
            continue
        s_movie.add(title)
        out_movies.write(f"{title}\n")

    out_movies.close()

    out_grade = open("./dataset/out_grade.csv", "w")
    out_grade.write("user_id,title,grade\n")

    # 从评分文件中提取评分信息
    files = ["./dataset/combined_data_1.txt"]
    for f in files:
        movie_id = -1
        for line in open(f, "r"):
            pos = line.find(":")
            if pos != -1:
                movie_id = int(line[:pos])
                continue
            line = line.strip().split(",")
            user_id = int(line[0])
            rating = int(line[1])

            if user_id > MAX_USER:
                continue

            if d_movie[movie_id].replace("\"", "") not in intersection_movie_set:
                continue

            out_grade.write(f"{user_id},{d_movie[movie_id]},{rating}\n")
    out_grade.close()


def TMDB():
    # pattern = re.compile("[A-Za-z0-9]+")
    out_genre = open("./dataset/out_genre.csv", "w", encoding='UTF-8')
    out_genre.write("title,genre\n")
    out_keyword = open("./dataset/out_keyword.csv", "w", encoding='UTF-8')
    out_keyword.write("title,keyword\n")
    out_productor = open("./dataset/out_productor.csv", "w", encoding='UTF-8')
    out_productor.write("title,productor\n")

    df = pd.read_csv("./dataset/tmdb_5000_movies.csv", sep=",")
    json_columns = ['genres', 'keywords', 'production_companies']
    for column in json_columns:
        df[column] = df[column].apply(json.loads)
    df = df[["genres", "keywords", "title", "production_companies"]]
    pre_sum = 0
    after_sum = 0
    for _, row in df.iterrows():
        title = row["title"]
        title = title.replace("\"", "")
        pre_sum += 1
        if title not in intersection_movie_set:
            continue
        after_sum += 1
        title = "\"" + title + "\""
        for g in row["genres"]:
            genre = g["name"]
            genre = "\"" + genre + "\""
            out_genre.write(f"{title},{genre}\n")

        for g in row["keywords"]:
            keyword = g["name"]
            keyword = "\"" + keyword + "\""
            out_keyword.write(f"{title},{keyword}\n")

        for g in row["production_companies"]:
            productor = g["name"]
            productor = productor.replace("\"", "")
            productor = "\"" + productor + "\""
            out_productor.write(f"{title},{productor}\n")
    print(f"pre_sum: {pre_sum}")
    print(f"after_sum: {after_sum}")

def get_intersection():
    netflix_dataframe = pd.read_csv("./dataset/netflix_movies.csv", encoding='ISO-8859-1')
    tmdb_dataframe = pd.read_csv("./dataset/tmdb_5000_movies.csv")
    netflix_movie_set = set()
    tmdb_movie_set = set()
    for item in netflix_dataframe['title']:
        netflix_movie_set.add(item)
    for item in tmdb_dataframe['title']:
        tmdb_movie_set.add(item)
    return netflix_movie_set.intersection(tmdb_movie_set)


if __name__ == "__main__":
    intersection_movie_set = get_intersection()
    Netflix()
    TMDB()


