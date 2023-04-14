'''
There are several parameters in the human detection and tracking script that can be tweak to adjust the execution speed.
Here are some of the key parameters:

scaleFactor: This parameter determines how much the image size is reduced at each image scale.
    A smaller scaleFactor will result in more image scales and more accurate detection but also slower processing.
    A larger scaleFactor will result in fewer image scales and faster processing but also less accurate detection.

minNeighbors: This parameter determines how many neighbors each candidate human rectangle should have to be
    considered a valid detection. A higher minNeighbors value will result in fewer false positives but also
    slower processing. A lower minNeighbors value will result in more false positives but also faster processing.

minSize: This parameter determines the minimum size of a human rectangle.
    A larger minSize value will result in faster processing but may also miss smaller humans.
    A smaller minSize value will result in slower processing but may detect smaller humans.

Frame resolution: Lowering the resolution of the camera can lead to faster processing speeds since there will be
    fewer pixels to process. You can set the frame resolution by using the cap.set() function as shown below:

Tweaking these parameters can have a significant impact on the accuracy and speed of the human
detection and tracking script. It's important to test the script thoroughly with different parameter values and
use cases to find the best configuration for the specific application.
'''

import cv2

# Load the pre-trained Haar cascade for human detection
human_cascade = cv2.CascadeClassifier('cascades/haarcascade_fullbody.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect humans in the grayscale frame
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected humans
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()