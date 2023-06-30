import re
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from flask import Flask, request, app, render_template
from tensorflow.keras import models

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image    

from werkzeug.utils import secure_filename

model=load_model(r"crime1.h5",compile=False)
app=Flask(__name__,template_folder='templates')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction')
def prediction():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        file = request.files['image']


        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        img= image.load_img(file_path, target_size=(64, 64))
        
        x=image.img_to_array(img) 
        x=np.expand_dims(x,axis=0)  
        pred=np.argmax(model.predict(x))
        op=['Fighting','Arrest','Vandalism','Assault','Stealing','Arson','NormalVideos','Abuse','Explosion','Robbery','Burglary','Shooting','Shoplifting','RoadAccidents']
        op1=['Fighting','Burglary','Vandalism','Assault','Stealing','RoadAccidents','NormalVideos','Explosion','Abuse','Robbery','Arrest','Shooting','Shoplifting','Arson']
        result = 'The predicted output is '+str(op[pred])          
    return render_template("predict.html", pred_text=result)  

if __name__ == '__main__':
    app.run(host='127.0.0.1')
