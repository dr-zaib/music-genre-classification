import requests
import sklearn.preprocessing as skp
import pandas as pd
from joblib import load


url = 'https://2imu17q39j.execute-api.eu-central-1.amazonaws.com/music-genre-classifier'

#data = {'url': 'http://bit.ly/mlbookcamp-pants'}


song = {
    'chroma_stft_mean': [0.4440117478370666],
    'chroma_stft_var': [0.0852029994130134],
    'rms_mean': [0.2037734985351562],
    'rms_var': [0.00659936433658],
    'spectral_centroid_mean': [2095.4208239067048],
    'spectral_centroid_var': [430313.48716756655],
    'spectral_bandwidth_mean': [2241.060906256248],
    'spectral_bandwidth_var': [110796.01228164523],
    'rolloff_mean': [4581.850948466502],
    'rolloff_var': [1717499.435882924],
    'zero_crossing_rate_mean': [0.0941514253431941],
    'zero_crossing_rate_var': [0.003422177139843],
    'harmony_mean': [-3.053571708733216e-05],
    'harmony_var': [0.0202102083712816],
    'perceptr_mean': [-0.001146224560216],
    'perceptr_var': [0.0124355517327785],
    'tempo': [135.99917763157896],
    'mfcc1_mean': [-74.18608856201172],
    'mfcc1_var': [2630.19384765625],
    'mfcc2_mean': [104.59842681884766],
    'mfcc2_var': [672.80517578125],
    'mfcc3_mean': [-8.379661560058594],
    'mfcc3_var': [544.7293090820312],
    'mfcc4_mean': [49.65111541748047],
    'mfcc4_var': [280.70770263671875],
    'mfcc5_mean': [-9.369983673095703],
    'mfcc5_var': [164.64053344726562],
    'mfcc6_mean': [21.44277572631836],
    'mfcc6_var': [137.95318603515625],
    'mfcc7_mean': [-16.32986831665039],
    'mfcc7_var': [144.6834259033203],
    'mfcc8_mean': [18.780681610107425],
    'mfcc8_var': [104.22476959228516],
    'mfcc9_mean': [-13.064760208129885],
    'mfcc9_var': [81.15306854248047],
    'mfcc10_mean': [18.465389251708984],
    'mfcc10_var': [67.09021759033203],
    'mfcc11_mean': [-14.948745727539062],
    'mfcc11_var': [63.98073196411133],
    'mfcc12_mean': [6.881842613220215],
    'mfcc12_var': [61.116844177246094],
    'mfcc13_mean': [-15.3265962600708],
    'mfcc13_var': [50.56884002685547],
    'mfcc14_mean': [4.77806282043457],
    'mfcc14_var': [62.86275863647461],
    'mfcc15_mean': [-11.097589492797852],
    'mfcc15_var': [52.935150146484375],
    'mfcc16_mean': [4.736111164093018],
    'mfcc16_var': [76.89418029785156],
    'mfcc17_mean': [-2.812528371810913],
    'mfcc17_var': [101.6149673461914],
    'mfcc18_mean': [6.798242092132568],
    'mfcc18_var': [107.96431732177734],
    'mfcc19_mean': [-5.803801536560059],
    'mfcc19_var': [103.87669372558594],
    'mfcc20_mean': [-3.767236709594727],
    'mfcc20_var': [92.24774932861328]
}

song_df = pd.DataFrame(song)
scaler = load('scaler.joblib')

X = scaler.transform(song_df)
X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
#X_reshaped = X_reshaped.astype('float32')

X_client = {'data': X_reshaped.tolist()}
#print(X_client)

result = requests.post(url, json=X_client).json()
print(result)