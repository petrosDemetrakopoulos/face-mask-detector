import os
import tensorflow as tf
import shutil
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle

print("The number of images with facemask:",len(os.listdir('data/with_mask')))
print("The number of images without facemask:",len(os.listdir('data/without_mask')))
IMG_HEIGHT = 150
IMG_WIDTH = 150

def train_test_split(source, trainPath, testPath, split_size):
	dataset = []
	for crnImage in os.listdir(source):
		data = source + '/' + crnImage
		if(os.path.getsize(data) > 0):
			dataset.append(crnImage)
	train_len = int(len(dataset) * split_size)
	test_len = int(len(dataset) - train_len)
	shuffled = random.sample(dataset, len(dataset))
	train = dataset[0:train_len]
	test = dataset[train_len:len(dataset)]
	print("train images with mask:",len(train))
	print("test images without mask:",len(test))

  #copying train and test images in seaparate directories
	for trainDataPoint in train: 
		crnTrainDataPath = source + '/' + trainDataPoint
		newTrainDataPath =  trainPath + '/' + trainDataPoint
		copyfile(crnTrainDataPath, newTrainDataPath)

	for testDataPoint in test:
		crnTestDataPath = source + '/' + testDataPoint
		newTestDataPath =  testPath + '/' + testDataPoint
		copyfile(crnTestDataPath, newTestDataPath)

model = tf.keras.models.Sequential([
    Conv2D(32, 3, activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.3), 
    Dense(256, activation='relu'),
    Dense(2, activation='softmax') # dense layer has a shape of 2 as we have only 2 classes 
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

acc = 0
val_loss = 0
val_acc = 0
loss = 0
val_loss = 0
epochs = 30

def trainModel():
  training_dir = "data/train"
  train_datagen = ImageDataGenerator(rescale=1.0/255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

  train_generator = train_datagen.flow_from_directory(training_dir, 
                                                      batch_size=10, 
                                                      target_size=(IMG_WIDTH, IMG_HEIGHT))
  validation_dir = "data/test"
  validation_datagen = ImageDataGenerator(rescale=1.0/255)
  validation_generator = validation_datagen.flow_from_directory(validation_dir, 
                                                           batch_size=10, 
                                                           target_size=(IMG_WIDTH, IMG_HEIGHT))
  checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')

  history = model.fit_generator(train_generator,
                                epochs=epochs,
                                validation_data=validation_generator,
                                callbacks=[checkpoint])
  global acc
  acc = history.history['accuracy']
  global val_acc
  val_acc = history.history['val_accuracy']
  global loss
  loss = history.history['loss']
  global val_loss
  val_loss = history.history['val_loss']

def plot():
  epochs_range = range(epochs)
  plt.figure(figsize=(6, 6))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

train_test_split('data/with_mask', 'data/train/training_with_mask', 'data/test/test_with_mask',0.8)
train_test_split('data/without_mask','data/train/training_without_mask', 'data/test/test_without_mask',0.8)
trainModel()
plot()

# Initializing webcam to live preview face mask detection ROIs on faces
# Seen this in this repo https://github.com/mk-gurucharan/Face-Mask-Detection
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0:'without mask',1:'with mask'}
color_dict = {0:(0,0,255),1:(0,255,0)}

size = 4
webcam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for (x, y, w, h) in faces:
        face_img = im[y:(y+h)*size, x:(x+w) * size]
        resized=cv2.resize(face_img,(IMG_WIDTH, IMG_HEIGHT))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized,(1,IMG_WIDTH,IMG_HEIGHT,3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        
        label = np.argmax(result,axis=1)[0]
      
        cv2.rectangle(im,(x* size,y* size),((x+w)* size,(y+h)* size),color_dict[label],2)
        cv2.rectangle(im,(x * size,(y* size)-40),((x+w)* size,y* size),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x* size + 10, (y* size)-10),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),2)
        
    cv2.imshow('Mask Detection', im)
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()