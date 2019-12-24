import cv2
import numpy as np

def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    name: "input" , shape: [1x3x256x456] - An input image in the format 
    [BxCxHxW], where:
    B - batch size
    C - number of channels
    H - image height
    W - image width. Expected color order is BGR.
    '''
    print(input_image.shape)
    
    image = np.copy(input_image)
    #print(preprocessed_image.shape)

    height = 256
    width = 456
        
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image


def text_detection(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    name: "input" , shape: [1x3x768x1280] - An input image in the format
    [BxCxHxW], where:

    B - batch size
    C - number of channels
    H - image height
    W - image width
    Expected color order - BGR.
    '''
    image = np.copy(input_image)
    print(image.shape)

    height = 768
    width = 1280
    
    
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image


def car_meta(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    
    ame: "input" , shape: [1x3x72x72] - An input image in following format 
    [1xCxHxW], where:
    - C - number of channels
    - H - image height
    - W - image width.
    '''
    image = np.copy(input_image)
    print(image.shape)

   
    height = 72
    width = 72
    
    
    image = cv2.resize(input_image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image
