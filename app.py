import numpy as np
import os
from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import tensorflow as tf

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

model = load_model(r"WCV.h5")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/home')
def home():
  return render_template('index.html')

@app.route('/input')
def input1():
  return render_template('input.html')

@app.route('/predict', methods = ["GET","POST"])
def res():
  if request.method == "POST":
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    print(basepath)
    filepath = os.path.join(basepath,'uploads',f.filename)
    f.save(filepath)

    img = image.load_img(filepath,target_size = (180,180,3))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)

    img_data = preprocess_input(x)
    prediction = np.argmax(model.predict(img_data),axis = 1)
    print(prediction)


    index = ['alien_test',  'cloudy','foggy','rainy','shine','sunrise']

    result = str(index[prediction[0]])
    print(result)
    return render_template('output.html',prediction = result)

if __name__ == "__main__":
  app.run(debug = True)