#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite
import numpy as np



interpreter = tflite.Interpreter(model_path='mugen-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


genres = [
    'classical',
    'rock',
    'metal',
    'country',
    'jazz',
    'blues',
    'reggae',
    'disco',
    'pop',
    'hiphop'
 ]



def predict(X):
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    genre_preds = dict(zip(genres, float_predictions))

    max_genre = max(genre_preds, key=genre_preds.get)
    max_value = genre_preds[max_genre]

    message = "your music's genre is: "

    return genre_preds, message, max_genre, max_value


def lambda_handler(event, context):
    data = event['data']

    data = np.array(data, dtype='float32')

    global_result, message, genre_result, genre_value = predict(data)
        
    # Consolidating results into a single dictionary
    response = {
        "global_result": global_result,
        "message": message,
        "genre_result": genre_result,
        "genre_value": genre_value
    }
    
    # Returning the response
    return response