import os
import flask
from PIL import Image
from tensorflow import keras
from flask import Flask , render_template  , request , send_file,  jsonify

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model(os.path.join(BASE_DIR , 'model.h5'))

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def load_image(filename):
    img = keras.preprocessing.image.load_img(filename, target_size=(128, 128))
    img = keras.preprocessing.image.img_to_array(img)
    img = img.reshape(-1, 128, 128, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img

def predictfunc(filename, model):
    # Load the Image
    img = load_image(filename)
    # Load Model
    # Predict the Class/Label
    result = model.predict(img)
    class_index = result.argmax(axis=-1)[0] # get the index of the max value in the result array

    flower = ""

    if class_index == 0:
        flower =  "Cattleya"
    elif class_index == 1:
        flower =  "Dendrobium"
    elif class_index == 2:
        flower =  "Oncidium"
    elif class_index == 3:
        flower =  "Phalaenopsis"
    elif class_index == 4:
        flower =  "Vanda"

    return flower

@app.route('/')
def home():
        
        return render_template("index.html")

@app.route('/success' , methods = ['POST'])
def success():

    if 'file' not in request.files:
        return 'No file uploaded', 400

    error = ''
    target_img = os.path.join(os.getcwd() , 'uploads')

    file = request.files['file']
    if file and allowed_file(file.filename):
        file.save(os.path.join(target_img , file.filename))
        img_path = os.path.join(target_img , file.filename)
        img = file.filename

        res_pred = predictfunc(img_path , model)

    data = {
        'flower' : res_pred,
        'success' : True,
        'message' : 'Register successfully',
    }

    if(len(error) == 0):
        return  jsonify(data), 200
    else:
        return "error bang"


if __name__ == "__main__":
    app.run(debug = True)


