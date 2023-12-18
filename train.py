
import numpy as np
import pandas as pd
import librosa 

import joblib
from joblib import dump

import sklearn 
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as skp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
plt.style.use('default')



# Feature importance function

def feature_importance_gradient(model, input_data):
    
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)  # Forward pass
        top_class = tf.argmax(predictions[0])
        top_output = predictions[:, top_class]
        
    gradients = tape.gradient(top_output, input_tensor)
    feature_importance = np.mean(np.abs(gradients), axis=0)
    
    return feature_importance


# Model creation function

def create_model(dropout_rate=0.1):
    
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    
    forward_layer = keras.layers.LSTM(64, dropout=dropout_rate, recurrent_dropout=dropout_rate)
    backward_layer = keras.layers.LSTM(64, go_backwards=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)
    model.add(keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer))
    model.add(keras.layers.Dropout(0.3))
    
    
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
              
    model.add(keras.layers.Dense(10, activation='softmax'))  
    
    adamopt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adamopt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model



# loading the data...

data_folder_path = 'Data/'
dataset = pd.read_csv(data_folder_path + 'features_30_sec.csv')


features = [
        'chroma_stft_mean', 'chroma_stft_var', 'rms_mean',
       'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var',
       'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
       'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
       'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
       'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
       'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
       'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
       'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
       'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
       'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
       'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
       'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
]

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


# mapping labels to numerical values

label_values = {
    'classical': 0, 
    'rock': 1,
    'metal': 2,
    'country': 3, 
    'jazz': 4, 
    'blues': 5, 
    'reggae': 6, 
    'disco': 7, 
    'pop': 8, 
    'hiphop': 9
}

dataset.label = dataset.label.map(label_values)


# getting rid of the features that are not really useful to the aim the of the project

dataset.drop(['filename', 'length'], axis=1, inplace=True)


# Shuffling and Spliting the dataset

dataset_shuffle = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

df_full_train, df_test = train_test_split(dataset_shuffle, test_size=0.2, random_state=1)
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.label.values
y_test = df_test.label.values

del df_full_train['label']
del df_test['label']

# data scaling

scaler = skp.StandardScaler()
X_train_scaled = scaler.fit_transform(df_full_train)
X_test_scaled = scaler.transform(df_test)

# Reshape the data for RNN input 
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# ...saving the scaler
try:
    # Save the scaler object using joblib
    dump(scaler, 'scaler.joblib')
    print('scaler saved properly!')
except Exception as e:
    # Handle exceptions that might occur during saving
    print("An error occurred while saving the scaler:", e)


# Training and saving the model

checkpoint = keras.callbacks.ModelCheckpoint(
    'mugen_model_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


dropout_rate=0.1


model = create_model(dropout_rate=dropout_rate)
history = model.fit(X_train_reshaped, y_full_train, validation_data=(X_test_reshaped, y_test), 
                    batch_size=32, epochs=250, callbacks=[checkpoint])

model_acc = model.evaluate(X_test_reshaped, y_test)
print('Overall model accuracy: ', model_acc[1])

best_acc = np.max(history.history['val_accuracy'])
print(f'Best validation accuracy: ', best_acc)

# Plotting accuracy and loss

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1,2,2)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')


preds = np.argmax(model.predict(X_test_reshaped), axis=1)


plt.figure(figsize=(16, 5))
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres);
disp.plot(cmap='Blues');
plt.title('Confusion Matrix for Validation Data')
plt.xticks(rotation=45)
plt.show()

print(classification_report(y_test, preds, target_names = genres))


# feature importance 


importance = feature_importance_gradient(model, X_test_reshaped)

importance = importance.flatten()

# Sort the indices by importance
sorted_indices = np.argsort(importance)
sorted_importance = importance[sorted_indices]
sorted_feature_names = [features[i] for i in sorted_indices]

# Plotting the feature importance
plt.figure(figsize=(8, 10)) 
plt.barh(range(len(sorted_importance)), sorted_importance, tick_label=sorted_feature_names, height=0.8, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Gradient-based Feature Importance')

plt.tight_layout(pad=1.0) 

plt.show()