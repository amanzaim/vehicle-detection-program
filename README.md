# vehicle-detection-program
This is a project done for a class(Digital Investigation). It uses the python language to detect, track and count the vehicles in the camera's view by using TensorFlow Object Detection API 

## Features

1. Detect vehicles using a camera
2. Predict the vehicle's size using the image area
3. Predict the vehicle's speed and direction using pixel locations
4. Get the color by using the color histogram

## Methodology


![Picture10](https://user-images.githubusercontent.com/29811408/133715612-5efb48b7-d619-40f1-b3d0-ca4214030cf4.png)

First, the program will read the input video frame by frame with OpenCV. Each frame is processed by “SSD with Mobilenet” model that was developed on TensorFlow. The model will detect multiple detection from the video frames. The detected vehicle image is used to prediction of the size using image area, this is generally a size prediction module in order to detect the size of the vehicle. The detected vehicle image also is used to color recognition by KNN is trained with color histogram, this module is to distinguish color in detected vehicle image frames. There is also detected vehicle image pixel location to predict the speed and directions using pixel location. After all detection passthrough the vehicle image will be detected in counting detected vehicle after the vehicle drive through the region of interest line. Lastly the video will be output with extract data from the counting, size, color, speed and direction of the vehicle.


## Implementation

Before using the application

![Picture11](https://user-images.githubusercontent.com/29811408/133715779-abd0dee0-ce90-46bc-8976-4530c9b8e967.png)


During the application

![Picture12](https://user-images.githubusercontent.com/29811408/133715831-55cfe477-f7f8-40ca-9ef6-2e3819b1eea4.png)


After the application run

![Picture13](https://user-images.githubusercontent.com/29811408/133715878-3fe2607e-0bfa-4861-951a-fc1228accce8.png)

## License

MIT Licensed
