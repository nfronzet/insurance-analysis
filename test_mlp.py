from __future__ import absolute_import, division, print_function

import keras
import numpy
import time

import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
cos_credentials = {
    'IAM_SERVICE_ID': '*****',
    'IBM_API_KEY_ID': '*****',
    'ENDPOINT': '*****',
    'IBM_AUTH_ENDPOINT': '*****',
    'BUCKET': '*****'
}

cos = ibm_boto3.client(service_name='s3',
    ibm_api_key_id=cos_credentials['IBM_API_KEY_ID'],
    ibm_service_instance_id=cos_credentials['IAM_SERVICE_ID'],
    ibm_auth_endpoint=cos_credentials['IBM_AUTH_ENDPOINT'],
    config=Config(signature_version='oauth'),
    endpoint_url=cos_credentials['ENDPOINT'])

#importing mnist dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

#selecting the first 1000 elements only
#train_labels = train_labels[:1000]
#test_labels = test_labels[:1000]

#reshaping elements for analysis
train_images = train_images[:len(train_images)].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:len(test_images)].reshape(-1, 28 * 28) / 255.0


# Returns a short sequential model
def create_model():
    model = keras.models.Sequential([
        keras.layers.Dense(512, activation=keras.activations.relu, input_shape=(784,)),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(32, activation=keras.activations.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=keras.activations.softmax)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])
  
    return model

# Create a basic model instance
model = create_model()
model.summary()

#Attempt to load pre-existing model parameters
print('Attempting to load previously saved weights...')
try:
    saved_weights = cos.get_object(Bucket=cos_credentials['BUCKET'], Key='weights.txt')['Body'].read().decode('utf-8')
    print('Previously saved weights loaded successfully')
except:
    print("No pre-existing model parameters found, creating new model...")
    
#Attempt to decode pre-existing parameters
try:
    w_list = []

    lines = saved_weights.split('\n')
    for line in lines:
        outer_arr = []
        vals = iter(line.split())
        for val in vals:
            if val == '[':
                inner_arr = []
                val = next(vals)
                while val != ']':
                    inner_arr.append(float(val))
                    val = next(vals)
                outer_arr.append(inner_arr)
            else:
                outer_arr.append(float(val))
        w_list.append(outer_arr)
    model.set_weights(w_list)
    print('Weights set successfully')
except:
    print('Unable to decode weights file')
    
#Untrained model
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained/pre-trained model, accuracy: {:5.2f}%".format(100*acc))

startTime = time.time()
model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels))
totTime = time.time() – startTime

#trained model
loss, acc = model.evaluate(test_images, test_labels)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))
print('Training time: ' + str(totTime) + ' seconds')

#Save model parameters to Cloud Storage
new_weights = model.get_weights()
w_buf = ''
print('Saving model weights to Cloud Storage...')
for list_element in new_weights:
    for d in list_element:
        if type(d) is numpy.ndarray:
            w_buf += '[ '
            for f in d:
                w_buf += str(f)
                w_buf += ' '
            w_buf += '] '
        else:
            w_buf += str(d)
            w_buf += ' '
    w_buf += '\n'

try:
    cos.put_object(Body=w_buf, Bucket=cos_credentials['BUCKET'], Key='weights.txt')
    print('Model weights saved successfully')
except Exception as e:
    print('File upload failed with error code: ' + str(e))
