import pandas
import pandas as pd


def analyse_netflix_tmdb_dataset():
    netflix_dataframe = pd.read_csv("./dataset/out_movies.csv", encoding='ISO-8859-1')
    tmdb_dataframe = pd.read_csv("./dataset/tmdb_5000_movies.csv")
    netflix_movie_set = set()
    tmdb_movie_set = set()
    for item in netflix_dataframe['title']:
        netflix_movie_set.add(item)
    for item in tmdb_dataframe['title']:
        tmdb_movie_set.add(item)
    # for item in tmdb_dataframe['original_title']:
    #     tmdb_movie_set.add(item)
    diff_movie = netflix_movie_set.difference(tmdb_movie_set)
    intersection_movie = netflix_movie_set.intersection(tmdb_movie_set)
    union_movie = netflix_movie_set.union(tmdb_movie_set)
    print(f"netflix movie: {len(netflix_movie_set)}")
    print(f"tmdb movie: {len(tmdb_movie_set)}")
    print(f"intersection movie: {len(intersection_movie)}")
    print(f"union movie: {len(union_movie)}")
    print(f"diff movie: {len(diff_movie)}")


def analyse_movie_json():
    movie_json_dataframe = pandas.read_json("./dataset/dbmovies.json", orient='record')
    movie_csv_dataframe = pandas.read_csv("./dataset/movies.csv")
    movie_rating_dataframe = pandas.read_csv("./dataset/ratings.csv")
    json_movie_set = set()
    csv_movie_set = set()
    csv_movie_title_id_dict = dict()
    rated_movie_id_set = set()
    for item in movie_json_dataframe['title']:
        json_movie_set.add(item)
    for index, item in movie_csv_dataframe.iterrows():
        # print(item)
        csv_movie_set.add(item['NAME'])
        csv_movie_title_id_dict[item['NAME']] = item['MOVIE_ID']
    for item in movie_rating_dataframe['MOVIE_ID']:
        rated_movie_id_set.add(item)
    intersection_movie = json_movie_set.intersection(csv_movie_set)
    intersection_movie_id = set()
    for item in intersection_movie:
        intersection_movie_id.add(csv_movie_title_id_dict[item])
    print(f"dbmovies.json movie: {len(json_movie_set)}")
    print(f"movies.csv movie: {len(csv_movie_set)}")
    print(f"intersection movie: {len(intersection_movie)}")
    print(f"intersection rated movie: {len(intersection_movie_id.intersection(rated_movie_id_set))}")


if __name__ == '__main__':
    # analyse_netflix_tmdb_dataset()
    analyse_movie_json()