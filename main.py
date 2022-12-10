from flask import Flask, render_template, request
from predict import ToxicClassifer
from utils import DHExchange
import json


app = Flask(__name__, template_folder='.')


@app.route('/', methods=['GET'])
def index():
    query = request.args.get('query')
    if query is None:
        query = ''
        answer = ['None', 'NaN']
    else:
        clf = ToxicClassifer()
        c_public = 197
        c_private = 199
        s_public = 151
        client = DHExchange(c_public, s_public, c_private)
        s_partial = client.generate_partial_key()
        c_full = client.generate_full_key(clf.get_client_partial_key(s_partial))
        filename = 'query.json'
        with open(filename, 'w+') as f:
            json.dump(query, f)
        filename_encrypted = client.encrypt_message(filename)
        filepath = clf.predict_from_json(s_partial, filename_encrypted)
        with open(filepath, 'r') as f:
            data = json.load(f)
        answer = ('toxic', float(data['probabilities'])) if data['probabilities'] > 0.5 \
            else ('not toxic', 1-float(data['probabilities']))

    return render_template(
        'templates/index.html',
        answer=answer[0],
        query=query,
        probs=answer[1]
    )


if __name__ == '__main__':
    app.run(debug=True)