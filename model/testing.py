import numpy as np
import pickle
from PIL import Image

def predictor():
  img = Image.open("test/drawing.png")
  print('image imported!')
  resized_img = np.array(img.resize((28,28)))
  img_arr = []
  for i in range(28):
    for j in range(28):
      img_arr.append(np.array(resized_img)[i][j][3])
  print('image processed!')

  model = pickle.load(open('model/image.pkl','rb'))

  y_pred = model.predict(np.array(img_arr).reshape(1,784))
  return y_pred[0]

print(f'Predicted number is {predictor()}.')