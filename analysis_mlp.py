import numpy as np
from matplotlib import pyplot as plt
import keras
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout
from cos_modelfiles import SaveLoad
import time

import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
cos_credentials = {'SERVICE_NAME': '******',
    'IBM_API_KEY_ID': '******',
    'IBM_AUTH_ENDPOINT': '******',
    'config': Config(signature_version=''******''),
    'ENDPOINT_URL': ''******'',
    'BUCKET': ''******''}

cos = ibm_boto3.client(service_name=cos_credentials['SERVICE_NAME'],
    ibm_api_key_id=cos_credentials['IBM_API_KEY_ID'],
    ibm_auth_endpoint=cos_credentials['IBM_AUTH_ENDPOINT'],
    config=cos_credentials['config'],
    endpoint_url=cos_credentials['ENDPOINT_URL'])

body = cos.get_object(Bucket='******',Key='insurance3_processed.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df = pd.read_csv(body)

#extract training and test sets
n_rows = len(df)
n_train = int(0.8*n_rows)

training_data = df.loc[0:n_train-1,'provincia_AG':'cilindrata'].values
training_labels = df.loc[0:n_train-1,'INC':'TUT']
test_data = df.loc[n_train:n_rows-1,'provincia_AG':'cilindrata'].values
test_labels = df.loc[n_train:n_rows-1,'INC':'TUT']

#define model
inputs = Input(shape=training_data.shape[1:])
common_branch = Dense(256, activation='relu', input_shape=training_data.shape[1:])(inputs)
common_branch = Dropout(0.2)(common_branch)
common_branch = Dense(128, activation='relu')(common_branch)
common_branch = Dropout(0.2)(common_branch)
common_branch = Dense(128, activation='relu')(common_branch)
common_branch = Dropout(0.2)(common_branch)
common_branch = Dense(64, activation='relu')(common_branch)
common_branch = Dropout(0.2)(common_branch)

inc_branch = Dense(1, activation='sigmoid', name='inc_output')(common_branch)
fur_branch = Dense(1, activation='sigmoid', name='fur_output')(common_branch)
esp_branch = Dense(1, activation='sigmoid', name='esp_output')(common_branch)
cri_branch = Dense(1, activation='sigmoid', name='cri_output')(common_branch)
evn_branch = Dense(1, activation='sigmoid', name='evn_output')(common_branch)
inf_branch = Dense(1, activation='sigmoid', name='inf_output')(common_branch)
ass_branch = Dense(1, activation='sigmoid', name='ass_output')(common_branch)
tut_branch = Dense(1, activation='sigmoid', name='tut_output')(common_branch)

model = Model(inputs=inputs, outputs=[inc_branch, fur_branch, esp_branch, cri_branch, evn_branch, inf_branch, ass_branch, tut_branch])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

#try to load model weights
try:
    weights = SaveLoad.cos_get_weights(cos, cos_credentials, 'ins_mlp_weights.txt')
    model.set_weights(weights)
except:
    print('Unable to load weights, retraining model')

#untrained model
test_labels_dict = {'inc_output': test_labels['INC'].values, 'fur_output': test_labels['FUR'].values, 'esp_output': test_labels['ESP'].values, 'cri_output': test_labels['CRI'].values, 'evn_output': test_labels['EVN'].values, 'inf_output': test_labels['INF'].values, 'ass_output': test_labels['ASS'].values, 'tut_output': test_labels['TUT'].values}
training_labels_dict = {'inc_output': training_labels['INC'].values, 'fur_output': training_labels['FUR'].values, 'esp_output': training_labels['ESP'].values, 'cri_output': training_labels['CRI'].values, 'evn_output': training_labels['EVN'].values, 'inf_output': training_labels['INF'].values, 'ass_output': training_labels['ASS'].values, 'tut_output': training_labels['TUT'].values}
eval_metrics = model.evaluate(test_data, test_labels_dict)
print("Untrained/pretrained model, accuracy (INC): {:5.2f}%,  accuracy (FUR): {:5.2f}%, accuracy (ESP): {:5.2f}%, accuracy (CRI): {:5.2f}%, accuracy (EVN): {:5.2f}%, accuracy (INF): {:5.2f}%, accuracy (ASS): {:5.2f}%, accuracy (TUT): {:5.2f}%".format(100*eval_metrics[9], 100*eval_metrics[10], 100*eval_metrics[11], 100*eval_metrics[12], 100*eval_metrics[13], 100*eval_metrics[14], 100*eval_metrics[15], 100*eval_metrics[16]))

#Train the model
start_time = time.time()

history = model.fit(training_data, training_labels_dict, validation_data=(test_data, test_labels_dict), epochs=200)

#Plot losses during training
plt.figure(0)
plt.title('Loss (INC)')
plt.plot(history.history['inc_output_loss'], label='Train (INC)')
plt.plot(history.history['val_inc_output_loss'], label='Test (INC)')
plt.legend()
plt.show()
plt.figure(1)
plt.title('Loss (FUR)')
plt.plot(history.history['fur_output_loss'], label='Train (FUR)')
plt.plot(history.history['val_fur_output_loss'], label='Test (FUR)')
plt.legend()
plt.show()
plt.figure(2)
plt.title('Loss (ESP)')
plt.plot(history.history['esp_output_loss'], label='Train (ESP)')
plt.plot(history.history['val_esp_output_loss'], label='Test (ESP)')
plt.legend()
plt.show()
plt.figure(3)
plt.title('Loss (CRI)')
plt.plot(history.history['cri_output_loss'], label='Train (CRI)')
plt.plot(history.history['val_cri_output_loss'], label='Test (CRI)')
plt.legend()
plt.show()
plt.figure(4)
plt.title('Loss (EVN)')
plt.plot(history.history['evn_output_loss'], label='Train (EVN)')
plt.plot(history.history['val_evn_output_loss'], label='Test (EVN)')
plt.legend()
plt.show()
plt.figure(5)
plt.title('Loss (INF)')
plt.plot(history.history['inf_output_loss'], label='Train (INF)')
plt.plot(history.history['val_inf_output_loss'], label='Test (INF)')
plt.legend()
plt.show()
plt.figure(6)
plt.title('Loss (ASS)')
plt.plot(history.history['ass_output_loss'], label='Train (ASS)')
plt.plot(history.history['val_ass_output_loss'], label='Test (ASS)')
plt.legend()
plt.show()
plt.figure(7)
plt.title('Loss (TUT)')
plt.plot(history.history['tut_output_loss'], label='Train (TUT)')
plt.plot(history.history['val_tut_output_loss'], label='Test (TUT)')
plt.legend()
plt.show()

#Trained model
eval_metrics = model.evaluate(test_data, test_labels_dict)
print("Trained model, accuracy (INC): {:5.2f}%,  accuracy (FUR): {:5.2f}%, accuracy (ESP): {:5.2f}%, accuracy (CRI): {:5.2f}%, accuracy (EVN): {:5.2f}%, accuracy (INF): {:5.2f}%, accuracy (ASS): {:5.2f}%, accuracy (TUT): {:5.2f}%".format(100*eval_metrics[9], 100*eval_metrics[10], 100*eval_metrics[11], 100*eval_metrics[12], 100*eval_metrics[13], 100*eval_metrics[14], 100*eval_metrics[15], 100*eval_metrics[16]))
print('Average accuracy: {:5.2f}%'.format(np.mean([100*eval_metrics[9], 100*eval_metrics[10], 100*eval_metrics[11], 100*eval_metrics[12], 100*eval_metrics[13], 100*eval_metrics[14], 100*eval_metrics[15], 100*eval_metrics[16]])))
print('Total training time: {:5.2f} seconds'.format(time.time()-start_time))

#try to save model weights
try:
    SaveLoad.cos_save_weights(model, cos, cos_credentials, 'ins_mlp_weights.txt')
except:
    print('File upload failed')
