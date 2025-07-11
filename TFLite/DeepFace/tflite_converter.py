from tensorflow import keras
from deepface.models.facial_recognition import Facenet
from tensorflow.keras.models import Model
import tensorflow as tf
import os

os.environ["DEEPFACE_HOME"] = "/home/work/Face/"

def load_and_add_finetune_layers():
    model = Facenet.load_facenet512d_model()
    model = Model(inputs=model.input, outputs=model.output)
    
    return model


def convert_keras_to_tflite(model_path, output_name, quantize=True, version=1):
    model = load_and_add_finetune_layers()
    model.load_weights(model_path)
    
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        filename = f'{output_name}_quant_v{version}.tflite'
    else:
        filename = f'{output_name}_tuned_v{version}.tflite'
    
    tflite_model = converter.convert()
    
    # Save the model.
    with open(filename, 'wb') as f:
        f.write(tflite_model)

        
if __name__ == "__main__":
    convert_keras_to_tflite('/home/work/Face/models/facenet_tf2/20250709-010156/ckpt_3500.weights.h5', 'facenet512_tuned')