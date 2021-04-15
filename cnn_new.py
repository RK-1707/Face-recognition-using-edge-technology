from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import pickle

x = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,
     3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,
     6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]

# Building the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape = (45, 50, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(32, 3, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(output_dim = 128, activation = 'relu')) 
model.add(Dense(output_dim = 1, activation = 'softmax'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/TrainingSet/',
                                                 target_size = (45, 50),
                                                 class_mode='binary')      #add batch_size according to dataset
 
test_set = test_datagen.flow_from_directory('dataset/TestSet/',
                                            target_size = (45, 50),
                                            class_mode='binary')          #add batch_size according to dataset

a = model.fit_generator(training_set,
                         samples_per_epoch = 118,
                         nb_epoch = 3,
                         validation_data = test_set,
                         nb_val_samples = 22)

print("\nAccuracy :", round((a.history.get('accuracy')[-1]) * 100), "%", "\nLoss :", round(a.history.get('loss')[-1]))

y_pred = model.predict(training_set)

print("\nConfusion Matrix :\n", confusion_matrix(x, y_pred))
print("\nClassification Report :\n", classification_report(x, y_pred))

model.summary()

# write the model to disk
f = open('Out/model', 'wb')
f.write(pickle.dumps(model))
f.close()