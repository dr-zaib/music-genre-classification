import tensorflow as tf 
from tensorflow import keras
import numpy as np


model = keras.models.load_model('mugen_model_v1_175_0.775.h5')


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
    
    preds = model.predict(X)

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
