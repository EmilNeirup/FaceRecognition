from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = [224, 224]

train_path = './data/train'
test_path = './data/test'

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
  layer.trainable = False
  
folders_path = train_path + '/*'  
folders = glob(folders_path)

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path, target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path, target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

print('Fitting model')
r = model.fit(training_set, validation_data=test_set, epochs=100, steps_per_epoch=len(training_set), validation_steps=len(test_set))

print('Done fitting model')
model.save('./models')
print('Model saved')
