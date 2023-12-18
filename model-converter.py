import tensorflow.lite as tflite
from tensorflow import keras



model_name = 'mugen_model_v1_175_0.775.h5'  # replace with the correct model name from your training
model = keras.models.load_model(model_name)

converter = tflite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tflite.OpsSet.TFLITE_BUILTINS, tflite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open('mugen-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)
