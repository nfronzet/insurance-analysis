import keras
from keras.layers import Dense, Dropout, LSTM
import time

#Import MNIST dataset
(X_train, y_train),(X_test, y_test) = keras.datasets.mnist.load_data()

#Normalize data
X_train = X_train/255.0
X_test = X_test/255.0

#One-hot encoding of output classes
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

#Define and compile recurrent model
model = keras.models.Sequential()

model.add(LSTM(64, input_shape=X_train.shape[1:], activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Untrained/pretrained model
loss, acc = model.evaluate(X_test, y_test)
print("Untrained/pretrained model, accuracy: {:5.2f}%".format(100*acc))

#Train the model
startTime = time.time()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4)
trainingTime = time.time() - startTime

#Trained model
loss, acc = model.evaluate(X_test, y_test)
print("Trained model, accuracy: {:5.2f}%".format(100*acc))
print("Total training time: " + str(trainingTime) + " seconds")
