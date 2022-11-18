from flask import Flask, render_template, request
from predict import get_prediction
import json


app = Flask(__name__, template_folder='.')


@app.route('/', methods=['GET'])
def index():
    query = request.args.get('query')
    if query is None:
        query = ''
        answer = ['None', 'NaN']
    else:
        filepath = get_prediction(query)
        with open(filepath, 'r') as f:
            json_items = json.load(f)
        answer = ('toxic', float(json_items['probalities'])) if json_items['probalities'] > 0.5 \
            else ('not toxic', 1-float(json_items['probalities']))

    return render_template(
        'templates/index.html',
        answer=answer[0],
        query=query,
        probs=answer[1]
    )


if __name__ == '__main__':
    app.run(debug=True)
