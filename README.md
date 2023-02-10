# object_detection_tensorflow_opencv

This project is implemented using cv2 framework and model is tensorflow2.* and deployed in Flask app (no UI)
Clone repo

run tensor.py

use below command to get the output in terminal:
curl http://127.0.0.1:8928/predict -X POST -H 'Content-Type: application/json' -d '{"publicUrl":"https://pyimagesearch.com/wp-content/uploads/2014/05/matplotlib-rgb-with-axis.jpg"}'
