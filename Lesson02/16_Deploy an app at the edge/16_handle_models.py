import cv2
import numpy as np


def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # TODO 1: Extract only the second blob output (keypoint heatmaps)
    #keypoint_heatmaps = output[1]
    # TODO 2: Resize the heatmap back to the size of the input
    #keypoint_heatmaps.
    
    print("____0_____", output.keys()) #122880
    print("____1_____", output['Mconv7_stage2_L1'].size) #69312
    print("____2_____", output['Mconv7_stage2_L1'].shape) #(1, 38, 32, 57) BxCxHxW
    print("____3_____", output['Mconv7_stage2_L2'].size) #34656
    print("____4_____", output['Mconv7_stage2_L2'].shape) #(1, 19, 32, 57) BxCxHxW
    
    print("____5_____", input_shape) #(750, 1000, 3) (h, w, BGR)

    out_pairwise_pixels = np.empty([output['Mconv7_stage2_L1'].shape[1], 750, 1000])
    #out_pairwise_pixels = np.empty([38, 750, 1000])
    out_heatmaps_pixels = np.empty([output['Mconv7_stage2_L2'].shape[1], 750, 1000])
    #for t in range(0,37):
    for t in range(0, output['Mconv7_stage2_L1'].shape[1]):
        out_pairwise_pixels[t] = cv2.resize(output['Mconv7_stage2_L1'][0][t], (1000, 750))
        
    for t in range(0,18):
        out_heatmaps_pixels[t] = cv2.resize(output['Mconv7_stage2_L2'][0][t], (1000, 750))
        
    return out_heatmaps_pixels


def handle_text(output, input_shape):
    '''
    Handles the output of the Text Detection model.
    Returns ONLY the text/no text classification of each pixel,
        and not the linkage between pixels and their neighbors.
    '''
    # TODO 1: Extract only the first blob output (text/no text classification)
    # TODO 2: Resize this output back to the size of the input
    
    print("____1_____", output['model/segm_logits/add'].size) #122880
    print("____2_____", output['model/segm_logits/add'].shape) #(1,2,192,320)
    print("____3_____", input_shape) #(667, 1000, 3) (h, w, BGR)
    
    text_classes = output['model/segm_logits/add']
    
    # TODO 2: Resize this output back to the size of the input
    print("____4____", text_classes.shape) #(1,2,192,320)
    print("____5____", len(text_classes[0])) # 2
    print("____6____", text_classes[0][0].shape) # (192,320)
    print("____6____", text_classes[0][1].shape) # (192,320)
    print("____7____", input_shape[0:2][::-1]) #(1000, 667)
    
    out_text_logits = np.empty([2, 667,1000])
    
    out_text_logits[0] = cv2.resize(text_classes[0][0], (1000,667))
    out_text_logits[1] = cv2.resize(text_classes[0][1], (1000,667))
#     for t in range(len(text_classes[0])): # 0..1
#         out_text_logits[t] = cv2.resize(text_classes[0][t], input_shape[0:2][::-1])
    
    return out_text_logits
    

def handle_car(output, input_shape):
    '''
    Handles the output of the Car Metadata model.
    Returns two integers: the argmax of each softmax output.
    The first is for color, and the second for type.
    '''
    # TODO 1: Get the argmax of the "color" output
    
    # TODO 2: Get the argmax of the "type" output
    
    return np.argmax(output['color']), np.argmax(output['type'])


def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "TEXT":
        return handle_text
    elif model_type == "CAR_META":
        return handle_car
    else:
        return None


'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
def preprocessing(input_image, height, width):
    '''
    Given an input image, and the model's height and width input image reqs
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image

def preprocessing_mine(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image