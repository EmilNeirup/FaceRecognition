
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os

model = load_model('./models')

face_cascade = cv2.CascadeClassifier('./data/extern/haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces == ():
        return None
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

image_to_predict = './images/predict'

for image in os.listdir(image_to_predict):
    image_path = os.path.join(image_to_predict, image)
    frame = cv2.imread(image_path)
    
    face=face_extractor(frame)

    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)
                     
        name="None matching"
        
        if(pred[0][0]>0.5 and pred[0][1]<0.5):
           name='Messi'

        if(pred[0][1]>0.5 and pred[0][0]<0.5):
           name='Ronaldo'

        print(name)
    else:
        print('No match found')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()