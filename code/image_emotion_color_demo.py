from keras.models import load_model
import numpy as np
import cv2

from Accessory_modules import preprocess_input
from Accessory_modules import apply_offsets

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'

emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

def process(frame):

    bgr_image = cv2.imread(frame,1)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    #this is standard syntax provided by opencv to detect faces
    faces=face_detection.detectMultiScale(gray_image, 1.3, 5)

    for face_coordinates in faces:

        #extend the suggested rectangular region containing face by little bit in all directions
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))

        except:
            continue

        gray_face = preprocess_input(gray_face, True)

        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        emotion_prediction = emotion_classifier.predict(gray_face)

        confidence_Array={}

        emotions={0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
         4: 'sad', 5: 'surprise', 6: 'neutral'}

        for i in range(0,7):
            confidence_Array[emotions[i]]=emotion_prediction[0][i]*100

        return confidence_Array

