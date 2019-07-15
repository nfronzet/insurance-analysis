from __future__ import absolute_import, division, print_function

import keras
import numpy
import time
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

def cos_get_weights(cos, credentials, filename):
    
    print('Attempting to load previously saved weights...')
    try:
        saved_weights = cos.get_object(Bucket=credentials['BUCKET'], Key=filename)['Body'].read().decode('utf-8')
        
        w_list = []

        lines = saved_weights.split('\n')
        lines.pop()
        
        for line in lines:
            tensor = []
            vals = iter(line.split())
            dims = []
            dim = int(next(vals))
            val = next(vals)
            while dim > 0:
                dims.append(int(val))
                val = next(vals)
                dim = dim - 1
            n = numpy.prod(dims)
            while n > 0:
                try:
                    tensor.append(float(val))
                    val = next(vals)
                    n = n - 1
                except:
                    break
            tensor = numpy.reshape(tensor,dims)
            w_list.append(tensor)
            
        print('Weights file found on Cloud Object Storage')
        return w_list
    except:
        raise FileNotFoundError()
        
def cos_save_weights(model, cos, credentials, filename):
    new_weights = model.get_weights()
    w_buf = ''
    print('Saving model weights to Cloud Storage...')
    for tensor in new_weights:
        dims = tensor.shape
        w_buf += str(len(dims))
        w_buf += ' '
        for dim in dims:
            w_buf += str(dim)
            w_buf += ' '
        length = numpy.prod(dims)
        t_arr = numpy.ndarray.reshape(tensor,length)
        for n in t_arr:
            w_buf += str(n)
            w_buf += ' '
        w_buf += ' \n'
    
    try:
        cos.put_object(Body=w_buf, Bucket=credentials['BUCKET'], Key=filename)
        print('Model weights saved successfully')
    except:
        raise ConnectionError()
        
        
# @hidden_cell
# The following code contains the credentials for a file in your IBM Cloud Object Storage.
# You might want to remove those credentials before you share your notebook.
cos_credentials = {
    'IAM_SERVICE_ID': '*****',
    'IBM_API_KEY_ID': '*****',
    'ENDPOINT': '*****',
    'IBM_AUTH_ENDPOINT': '*****',
    'BUCKET': '*****',
}

cos = ibm_boto3.client(service_name='s3',
    ibm_api_key_id=cos_credentials['IBM_API_KEY_ID'],
    ibm_service_instance_id=cos_credentials['IAM_SERVICE_ID'],
    ibm_auth_endpoint=cos_credentials['IBM_AUTH_ENDPOINT'],
    config=Config(signature_version='oauth'),
    endpoint_url=cos_credentials['ENDPOINT'])

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

#Reshape data to fit model
X_train = X_train.reshape(len(X_train),28,28,1)
X_test = X_test.reshape(len(X_test),28,28,1)

#One-hot encoding of output classes
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

#Create convolutional model
model = Sequential()
model.add(Conv2D(28, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(14, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Attempt to load previously saved weights
try:
    weights = cos_get_weights(cos, cos_credentials,'CNNweights.txt')
    model.set_weights(weights)
except:
    print('Exception encountered during loading procedure: weights file does not exist or is not compatible with this model')

#Untrained/pretrained model
loss, acc = model.evaluate(X_test, y_test)
print("Untrained/pretrained model, accuracy: {:5.2f}%".format(100*acc))

#Train the model
startTime = time.time()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
trainingTime = time.time() - startTime

#Trained model
loss, acc = model.evaluate(X_test, y_test)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))
print("Total training time: " + str(trainingTime) + " seconds")

#Attempt to save weights to Cloud Storage
try:
    cos_save_weights(model, cos, cos_credentials, 'CNNweights.txt')
except:
    print('File upload failed')
