from flask import Flask
import main

app = Flask(__name__)


@app.route('/recommend')
def recommend():
    return main.recommend()


if __name__ == '__main__':
    app.run()
