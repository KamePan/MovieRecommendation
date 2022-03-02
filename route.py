from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import main
app = Flask(__name__)

CORS(app, resources=r'/*')


@app.route('/recommend')
def recommend():
    args = request.args
    embed = args.get('embed')
    recommend = args.get('recommend')
    uid = args.get('uid')
    print(f"{embed} {recommend} {uid}")
    predict_list = main.recommend_merge_top_k(user_id=uid, recommend_num=10)
    result = list()
    for item in predict_list:
        item_dict = dict()
        item_dict["Title"] = item[0]
        item_dict["Grade"] = item[1]
        item_dict["Recommender"] = item[2]
        if len(item[3]) == 0:
            item_dict["Genre"] = "暂无类别"
        else:
            item_dict["Genre"] = ", ".join(item[3])
        result.append(item_dict)
    return jsonify(result)


if __name__ == '__main__':
    app.run()
