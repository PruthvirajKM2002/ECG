from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

x_train = train_datagen.flow_from_directory("C:\\Users\\Lenovo\\Desktop\\Jupyter\\FinalYearProject\\data\\train",target_size = (64,64),batch_size = 32,class_mode = "categorical")
x_test = test_datagen.flow_from_directory("C:\\Users\\Lenovo\\Desktop\\Jupyter\\FinalYearProject\\data\\test",target_size = (64,64),batch_size = 32,class_mode = "categorical")


model = Sequential()

model.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = "relu"))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # ANN Input...

#Adding Dense Layers



model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 6,kernel_initializer = "random_uniform",activation = "softmax"))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(generator=x_train,steps_per_epoch = len(x_train), epochs=10, validation_data=x_test,validation_steps = len(x_test))

model.save('ECG.h5')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the siamese model
model = load_model('ECG.h5')

# Load and preprocess the unknown image
img = image.load_img("C:\\Users\\Lenovo\\Desktop\\Jupyter\\FinalYearProject\\fig_37.png", target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Duplicate the image to create a pair
x_pair = [x, x]

# Predict with the model
pred = model.predict(x_pair)

# Get the predicted class
y_pred = np.argmax(pred)
print("Predicted class:", y_pred)

img=image.load_img("C:\\Users\\Lenovo\\Desktop\\Jupyter\\FinalYearProject\\fig_37.png",target_size=(64,64))
x=image.img_to_array(img)
import numpy as np
x=np.expand_dims(x,axis=0)
pred = model.predict(x)
y_pred=np.argmax(pred)
y_pred

index=['left Bundle Branch block',
       'Normal',
       'Premature Atrial Contraction',
       'Premature Ventricular Contraction',
       'Right Bundle Branch Block',
       'Ventricular Fibrillation']
result = str(index[y_pred])
result
