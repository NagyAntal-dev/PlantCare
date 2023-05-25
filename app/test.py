import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

def loadLabels():
    #ToDo:
    # 1. Relative Path for Labels
    return [ item.strip() for item in open('C:\\Users\\Nagy Antal\\Desktop\\tf\\app\\labels.txt','r').readlines() ]

image =  tf.image.resize(Image.open("C:\\Users\\Nagy Antal\\Desktop\\tf\\app\\daisy.jpg"), (224,224)) / 255.0
image = np.array(image)[None,:,:]
image[0] = None
print(image.shape)

labels = loadLabels()

model = tf.keras.models.load_model(
    "models\BOARDED_oxfordflower_OUR_1683656864",
    custom_objects={'KerasLayer': hub.KerasLayer})

'''for layer in model.layers:
    print(f"name = {layer.name}, params = {layer.count_params():d}")
    if layer.count_params() > 0:
        destLayers = [x for x in model.layers if x.name == layer.name]
        if (len(destLayers) == 1) and (destLayers[0].count_params() == layer.count_params()):
            destLayers[0].set_weights(layer.get_weights())
            print(f'Súlyozás másolva {destLayers[0].name}')'''

res = model.predict([image])
print(tf.argmax(res))
