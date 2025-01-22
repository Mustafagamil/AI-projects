# Flask
import joblib as joblib
from flask import Flask, request, render_template, url_for, jsonify
import tensorflow as tf
import os
import numpy as np
from keras.optimizers import Adam
from tensorflow.keras import models
from PIL import Image

#scaler = joblib.load('scaler.pkl')
from tensorflow.python.keras.models import load_model

classes = ['0','1']



def preprossing(image):
    image=Image.open(image)
    image = image.resize((50, 50))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 50, 50, 3)
    return image_arr


print("Loading Model ...")


model= models.load_model('Brest CNN (1).h5  ', compile=False)
model.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
print("Model Loaded")
app = Flask(__name__)


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)

        result = model.predict(image_arr)
        print(result)
        ind = np.argmax(result)

        prediction = str(ind)
        return jsonify({'prediction': prediction})





if __name__ == '__main__':
    app.run(debug=True)