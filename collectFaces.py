import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('./data/extern/haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces == ():
        return None

    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face

source_images_dir = './data/raw/messi'
training_images_dir = './data/train/messi/'

count = 0

for image in os.listdir(source_images_dir):
    image_path = os.path.join(source_images_dir, image)
    frame = cv2.imread(image_path)
    
    if face_extractor(frame) is not None :
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        file_name_path = training_images_dir + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        print("Face found")
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()      
print("Collecting Samples Complete")