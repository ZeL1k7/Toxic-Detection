from flask import Flask, render_template, request
from predict import get_prediction


app = Flask(__name__, template_folder='.')


@app.route('/', methods=['GET'])
def index():
    query = request.args.get('query')

    if query is None:
        query = ''
        probs = 0,
        answer = 'no'
    else:
        probs = get_prediction(query)
        answer = ('toxic', float(probs)) if probs > 0.5 else ('not toxic', 1-float(probs))
    return render_template(
        'templates/index.html',
        answer=answer[0],
        query=query,
        probs=answer[1]
    )


if __name__ == '__main__':
    app.run(debug=True)
