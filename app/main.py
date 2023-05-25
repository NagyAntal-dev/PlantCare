import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub

def loadLabels():
    #ToDo:
    # 1. Relative Path for Labels
    return [ item.strip() for item in open('C:\\Users\\Nagy Antal\\Desktop\\tf\\app\\labels.txt','r').readlines() ]

def FlowerModel(file):
    model = tf.keras.models.load_model(
    "models\BOARDED_oxfordflower_BiT_1683555554",
    custom_objects={'KerasLayer': hub.KerasLayer})
    
    
    
    return f'Hello {type(file)}!'

demo = gr.Interface(fn=FlowerModel, inputs="image", outputs="text", live=True)

demo.queue(concurrency_count=5, max_size=20).launch()