import requests
import json
import cv2
import glob

#api url- the url of machine on which web.py is running
addr = 'http://192.168.0.6:8080'
test_url = addr

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

#for single image
#img = cv2.imread('uploads/happyman.jpg')

#put the path of image here
for image in glob.iglob("uploads/*"):
    print(image)
    img = cv2.imread(image)


    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)

    # send http request with image and receive response
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)

    # decode response
    print (json.loads(response.text))

#json response like
#{'neutral': 17.60459691286087, 'sad': 58.58122706413269, 'angry': 7.700709253549576, 'surprise': 0.8887623436748981, 'disgust': 0.09097328875213861, 'fear': 5.724337697029114, 'happy': 9.409397095441818}