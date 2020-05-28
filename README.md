PROJECT SPECIFICATION
Deploy a People Counter App at the Edge

This project was very helpful to understand deep learning object detection using Intel OpenVINO Toolkit. 

I found it extremely helpful for accelerating the project development time.

However, I should say that a marketable people counter app should deal with model accuracy by augmenting the plain detection with object tracking to avoid false negatives that increments the people count incorrectly.

Another option to avoid counting persons repeatedly for low confidence detectors or detectors where no instance detection is available is to use

​ a) skimage.metrics the structural_similarity frame comparator in order to discard counting people from very similar frames.

​ b) scipy.spatial.distance.cdist to compare prediction boxes between N consecutive frames to discard repeated box counts in predictions.

Although I tried to install scipy and skimage it in the workspace I didn't succeded.

Regarding model optimizer, I found the process of transforming models that were not pretrained by Intel quiet hard. I think that the app developer should have detailed knowledge of the model in order to convert it properly and successfully. Examples:

​ a) faster_rcnn_resnet101_coco_2018_01_28 although converted successfully with model optimizer, it requires special case when handling input shape in the people counter python app. See MODELS.md for further details.

​ b) MobileNetSSD_deploy.caffemodel. although it was converted successfully with model optimizer when loading the model with inference engine throws run-time error “Weights/biases are empty for layer: conv0/bn used in MKLDNN node: conv0/bn”

One more nice to have is a unit test suite to support the coding activity.

Despite all previous drawbacks, overall experience was very engaging and satisfactory.
