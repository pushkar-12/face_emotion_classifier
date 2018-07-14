import cv2
import pandas as pd
import numpy as np


def get_data_from_fer2013(image_size=(48,48),dataset_path="data/fer2013.csv"):

    df = pd.read_csv(dataset_path)
    #now df is a dataframe that has three columns 'emotion', 'pixels', and 'Usage'

    #Now we are looking at 'pixels' column of the dataframe
    pixels = df['pixels'].tolist()
    #now pixels becomes a list of strings with each string representing
    #a single face image of size 48*48. Each string contains 48*48=2304 pixel values
    #separated by spaces

    width, height = image_size

    all_faces = []
    for pixel_sequence in pixels:#pixel_sequence is a string representing one face image

        one_face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        #now one_face is an array holding 2304 pixel values

        one_face = np.asarray(one_face).reshape(width, height)
        one_face = cv2.resize(one_face.astype('uint8'), image_size)

        all_faces.append(one_face.astype('float32'))

    #all_faces becomes a list of 48*48 shaped np arrays
    all_faces = np.asarray(all_faces)

    all_faces = np.expand_dims(all_faces, -1)

    #each row in dataset contains one emotion value in range [0,6] for each face
    #pd.get_dummies() converts this single integer into a row of 7 integers
    #with 1 set for the present emotion and 0 for others
    emotions = pd.get_dummies(df['emotion']).as_matrix()

    return all_faces, emotions

def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data

