import os
import tensorflow as tf

from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm

def set_env():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    
def load_model_from_ckpt(load_path, save_path, sz=160, backbone='ResNet50', is_train=False):
    model = ArcFaceModel(size=sz, backbone_type=backbone, training=is_train)
    ckpt_path = tf.train.latest_checkpoint(load_path)
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        model.save(save_path, save_format="tf")
    return model


def TF_Lite_Convert(saved_path, to_save_path, *ops):
    # Convert the SavedModel to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("/home/work/Face/arcface-tf2/checkpoints2/arcface_res50_3.pb")
    converter.experimental_enable_resource_variables = True
    converter.target_spec.supported_ops = ops[0]
    tflite_model = converter.convert()
    # Save to file
    with open(to_save_path, "wb") as f:
        f.write(tflite_model)

        
if __name__=="__main__":
    set_env()
    pbfile = "/home/work/Face/arcface-tf2/checkpoints2/arcface_res50_3.pb"
    model = load_model_from_ckpt('/home/work/Face/arcface-tf2/checkpoints/arc_res50', pbfile)
    TF_Lite_Convert(pbfile, 
                    "/home/work/Face/arcface-tf2/checkpoints2/arcface_res50_v3.tflite",
                    [tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
                     tf.lite.OpsSet.SELECT_TF_OPS    # enable TensorFlow ops.
                    ]
                   )
    
    