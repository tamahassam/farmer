from farmer import app
from farmer.ImageAnalyzer import fit
from flask import request, make_response, jsonify
from keras.models import load_model
from configparser import ConfigParser
import cv2
import os
import numpy as np
import shutil


@app.route('/train', methods=["POST"])
def train():
    form = request.json
    form = {k: v for (k, v) in form.items() if v}
    parser = ConfigParser()
    parser['project_settings'] = form
    fit.train(parser)
    return make_response(jsonify(dict()), 200)


@app.route('/predict', methods=["POST"])
def predict():
    img_file = request.files['image']
    file_data = img_file.stream.read()
    nparr = np.fromstring(file_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result_dir = request.files['result_dir'].stream.read()
    model_path = os.path.join(
        result_dir.decode('utf-8'),
        'model',
        'best_model.h5'
    )
    model = load_model(model_path)
    input_img = np.expand_dims(img, axis=0)/255
    predictions = model.predict(input_img)[0]
    predictions = [float(prediction) for prediction in predictions]
    return make_response(jsonify(dict(prediction=predictions)))


@app.route('/test', methods=["POST"])
def test():
    form = request.json
    form = {k: v for (k, v) in form.items() if v}
    parser = ConfigParser()
    parser['project_settings'] = form
    model_path = os.path.join(
        form["result_dir"],
        'model',
        'best_model.h5'
    )
    parser['project_settings']['model_path'] = model_path
    report = fit.evaluate(parser)
    return make_response(jsonify(report))


@app.route('/delete_model', methods=["POST"])
def delete_model():
    form = request.json
    shutil.rmtree(form.get('result_dir'))
    return make_response('', 202)
