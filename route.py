from flask import Flask, jsonify
from flask_cors import CORS
import main

app = Flask(__name__)

CORS(app, resources=r'/*')


@app.route('/recommend')
def recommend():
    predict_list = main.recommend()
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
