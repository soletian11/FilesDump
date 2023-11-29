import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np

#url='https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'

intrepreter_hw=tflite.Interpreter(model_path='/home/ubuntu/FilesDump/bees-wasps.tflite')
intrepreter_hw.allocate_tensors()
inputhw_index=intrepreter_hw.get_input_details()[0]['index']
outputhw_index=intrepreter_hw.get_output_details()[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    return x / 255.0

def predict(url):
    img=download_image(url)
    img=prepare_image(img,(150,150))
    x=np.array(img, dtype='float32')
    X=preprocess_input(np.array([x]))
    print(X[0,0,0,0])
    print("output index" ,outputhw_index)
    intrepreter_hw.set_tensor(inputhw_index,X)
    intrepreter_hw.invoke()
    preds=intrepreter_hw.get_tensor(outputhw_index)
    return preds

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

# predict(url)