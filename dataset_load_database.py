import pandas
from neo4j import GraphDatabase


def process_dataset():
    """
    处理包含电影元数据信息的 json 文件以及评分数据 csv 文件
    """
    movie_metadata_dataframe = pandas.read_json('./dataset/dbmovies.json', orient='record')
    metadata_movie_set = set()

    """
    对于 json 中份元数据信息，将电影title和其他电影内涵实体关系导入csv文件
    """
    out_movie_new = open('./dataset/out_movie_new.csv', 'w')
    out_director_new = open('./dataset/out_director_new.csv', 'w')
    out_actor_new = open('./dataset/out_actor_new.csv', 'w')
    out_district_new = open('./dataset/out_district_new.csv', 'w')
    out_category_new = open('./dataset/out_category_new.csv', 'w')
    out_language_new = open('./dataset/out_language_new.csv', 'w')
    out_movie_new.write('title\n')
    out_director_new.write('title,director\n')
    out_actor_new.write('title,actor\n')
    out_district_new.write('title,district\n')
    out_category_new.write('title,category\n')
    out_language_new.write('title,language\n')

    for index, item in movie_metadata_dataframe.iterrows():
        title = item['title'].replace('\"', ' ')
        title = '\"' + title + '\"'
        directors = item['director']
        actors = item['actor']
        districts = item['district']
        categories = item['category']
        languages = item['language']

        metadata_movie_set.add(title)

        out_movie_new.write(f'{title}\n')
        # 需要 id 吗，按道理是需要，但是我的实现里用的是 neo4j 的隐含 id，到时候返回推荐的 title，再用 title 查 mysql 表就行
        # python 这边利用知识图谱给出推荐结果，再通过 java 端查询 mysql 数据库给出元数据信息展示

        if directors is not None:
            for director in directors:
                out_director_new.write(f'{title},\"{director}\"\n')

        if actors is not None:
            for actor in actors:
                out_actor_new.write(f'{title},\"{actor}\"\n')

        if directors is not None:
            for district in districts:
                out_district_new.write(f'{title},\"{district}\"\n')

        if categories is not None:
            for category in categories:
                out_category_new.write(f'{title},\"{category}\"\n')

        if languages is not None:
            for language in languages:
                out_language_new.write(f'{title},\"{language}\"\n')

    """
    提取电影评分信息 (user, title, grade)
    """
    movie_dataframe = pandas.read_csv('./dataset/movies.csv')
    rating_dataframe = pandas.read_csv('./dataset/ratings.csv')
    rating_movie_dict = dict()
    rating_movie_set = set()
    out_grade_new = open('./dataset/out_grade_new.csv', 'w')
    out_grade_new.write('user_id,title,grade\n')
    for index, item in movie_dataframe.iterrows():
        rating_movie_dict[item['MOVIE_ID']] = item['NAME']
        rating_movie_set.add(item['NAME'])
    user_md5_id_dict = dict()
    user_id_auto_increment = 0
    rating_cnt_before = 0
    rating_cnt_after = 0
    not_rating_cnt = 0
    for index, item in rating_dataframe.iterrows():
        rating_movie_id = item['MOVIE_ID']
        '''滤除有评分但没有在电影集里的电影'''
        if rating_movie_id not in rating_movie_dict.keys():
            not_rating_cnt += 1
            continue
        rating_movie_title = rating_movie_dict[rating_movie_id].replace('\"', ' ')
        rating_movie_title = '\"' + rating_movie_title + '\"'
        '''滤除没有在元数据电影集里的电影'''
        rating_cnt_before += 1
        if rating_movie_title not in metadata_movie_set:
            continue
        rating_cnt_after += 1
        rating_movie_grade = item['RATING']
        rating_movie_user_md5 = item['USER_MD5']
        if rating_movie_user_md5 not in user_md5_id_dict.keys():
            user_md5_id_dict[rating_movie_user_md5] = user_id_auto_increment
            user_id_auto_increment += 1
        out_grade_new.write(f'{user_md5_id_dict[rating_movie_user_md5]},{rating_movie_title},{rating_movie_grade}\n')
    print(f"rating_cnt_before: {rating_cnt_before}")
    print(f"rating_cnt_after: {rating_cnt_after}")
    print(f"not_rating_cnt: {not_rating_cnt}")


def load_meta_data():
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "pan151312"), max_connection_lifetime=200)
    session = driver.session()

    '''删除数据库中原有的实体和关系'''
    session.run("""MATCH ()-[r]->() DELETE r""")
    session.run("""MATCH (r) DELETE r""")

    '''将 csv 文件中的电影、类型、演员、导演、地区数据加载到 neo4j 中'''
    print("Loading movies...")
    session.run("""
            LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_movie_new.csv" AS csv
            CREATE (:Movie {title: csv.title})        
        """)

    print("Loading categories...")
    session.run("""
            LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_category_new.csv" AS csv
            CREATE (m:Movie {title: csv.title})
            CREATE (c:Category {category: csv.category})
            MERGE (m)-[:HAS_CATEGORY]->(c)        
        """)

    print("Loading actors...")
    session.run("""
            LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_actor_new.csv" AS csv
            CREATE (m:Movie {title: csv.title})
            CREATE (a:Actor {name: csv.actor})
            MERGE (a)-[:ACT_IN]->(m)        
        """)

    print("Loading directors...")
    session.run("""
            LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_director_new.csv" AS csv
            CREATE (m:Movie {title: csv.title})
            CREATE (d:Director {name: csv.director})
            MERGE (m)-[:DIRECTED_BY]->(d)        
        """)

    print("Loading districts...")
    session.run("""
            LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_district_new.csv" AS csv
            CREATE (m:Movie {title: csv.title})
            CREATE (d:District {district: csv.district})
            MERGE (m)-[:FILMED_IN]->(d)        
        """)

    print("Loading languages...")
    session.run("""
            LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_language_new.csv" AS csv
            CREATE (m:Movie {title: csv.title})
            CREATE (l:Language {language: csv.language})
            MERGE (m)-[:HAS_LANGUAGE]->(l)        
        """)


def load_rating_data():
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "pan151312"))
    session = driver.session()

    '''删除数据库中原有的用户实体和评分关系'''
    session.run("""MATCH ()-[r:RATED]->() DELETE r""")
    session.run("""MATCH (u:User) DELETE u""")

    print("Loading gradings...")
    session.run("""
            LOAD CSV WITH HEADERS FROM "file:///var/lib/neo4j/import/movie_dataset/out_grade_new.csv" AS csv
            CREATE (m:Movie {title: csv.title})
            CREATE (u:User {id: toInteger(csv.user_id)})
            MERGE (u)-[:RATED {grading: toInteger(csv.grade)}]->(m)        
        """)


if __name__ == '__main__':
    # process_dataset()
    load_meta_data()
