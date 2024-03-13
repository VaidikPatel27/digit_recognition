import io
from flask import Flask, render_template, request
import numpy as np
import pickle
from PIL import Image
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/digit_recognization',methods=['POST'])

def img_show():

    image_data_url = request.json['imageDataURL']
    image_data = base64.b64decode(image_data_url.split(',')[1])
    # with open('image.png', 'wb') as f:
    #     f.write(image_data)
    image_stream = io.BytesIO(image_data)
    img = Image.open(image_stream)
    resized_img = np.array(img.resize((28, 28)))
    img_arr = []
    for i in range(28):
        for j in range(28):
            img_arr.append(np.array(resized_img)[i][j][3])
    model = pickle.load(open('model/image.pkl', 'rb'))
    y_pred = model.predict(np.array(img_arr).reshape(1, 784))
    return str(y_pred[0])

if __name__ == '__main__':
    app.run(host = '0.0.0.0' , port = 8100 , debug=True)