
import numpy as np
import cv2
import pickle 
import json 
from keras.models import load_model 
from PIL import Image 
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
from flask import Flask,request,jsonify,abort 


app = Flask(__name__)
model_path = "Model/model3-054.h5"

@app.route('/')
def index():
    return_data={
        "data" :  "malar-Ai",
    }
    return app.response_class(response=json.dumps(return_data),mimetype='application/json') 

def preProcess_img(img_file):
    try :
        image = cv2.imdecode(np.fromstring(img_file.read(),np.uint8),cv2.IMREAD_UNCHANGED)
        image = Image.fromarray(image,'RGB')
        image = np.array(image.resize((224,224)))
        image = image/255
        final_image = []
        final_image.append(image)
        final_image = np.array(final_image)
        return (True,final_image)
    except  Exception as e :
        print(e)
        return (False,str(e))

@app.route('/classify',methods=['POST'])
def classify_malaria_cells():
    try :
         if ("file" in request.files and request.files['file']  is not None) :
             img = request.files['file']
             is_successful,preProcessed_image = preProcess_img(img)
             if (is_successful) :
                 malaria_model = load_model(model_path)
                 score = malaria_model.predict(preProcessed_image)
                 label_index = np.argmax(score)
                 classification = "Uninfected" if label_index==0 else "Infected"
                 max_score = round(np.max(score),2)*100
                 s = str(max_score)
                 return_data = {
                     "error" : "0",
                     "message" : "Successful",
                     "classification" : classification,
                     "probability" : s ,
                  }
             else:
                 return_data = {
                     "error" : "1",
                     "message" : "Image Processing Error"
                 }
         else :
              return_data = {
                     "error" : "2",
                     "message" : "Invalid Parameters"
                 }
    except Exception as e :
         return_data = {
                     "error" : "3",
                     "message" : f"[Error] : {e}"
                 }
    return app.response_class(response=json.dumps(return_data),mimetype='application/json')          
       

if __name__ == "__main__" :
    app.run(port=8080,debug=False,threaded=False)

