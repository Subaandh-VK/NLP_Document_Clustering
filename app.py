import pandas as pd
from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)

# Import the model
model = load('models/k_means_clustering.pkl')
vectorizer = load('models/tfidf_vectorizer.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    form_dict = request.form.to_dict()
    words = []

    if request.files['json_file']:
        print(request.files.values())
        df = pd.read_json(request.files['json_file'])
        clause_list = df['clauses'].to_list()
        clause_content = ''.join(str(val) for val in clause_list)
        print(clause_content)

        x = vectorizer.transform([clause_content])
        prediction = model.predict(x)

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        feature_names = vectorizer.get_feature_names()
        for j in order_centroids[prediction[0], :10]:
            print('Feature names ', feature_names[j], end=' ')
            words.append(feature_names[j])

        prediction = 'Prediction for document is cluster:' + str(prediction[0])

    elif form_dict.get('file_content') != '':
        x = vectorizer.transform([form_dict.get('file_content')])
        prediction = model.predict(x)

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        feature_names = vectorizer.get_feature_names()
        for j in order_centroids[prediction[0], :10]:
            print('Feature names ', feature_names[j], end=' ')
            words.append(feature_names[j])

        prediction = 'Prediction for text is cluster:' + str(prediction[0])

    else:
        prediction = 'Unable to predict'

    words = ' '.join(word for word in words)

    return render_template('index.html', prediction=f" Result : {prediction}", words=f"Top words: \n {words}")


if __name__ == '__main__':
    app.run(port=5001)
