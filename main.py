from flask import Flask
from flask_restful import Api
from predict import ToxicClassifer

app = Flask(__name__, template_folder='.')
api = Api(app)


api.add_resource(ToxicClassifer, "/toxic")
if __name__ == '__main__':
    app.run(debug=True)
