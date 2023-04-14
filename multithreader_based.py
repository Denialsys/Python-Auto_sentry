'''
It's possible to leverage the GPU of a computer to accelerate the processing of
computer vision tasks in Python. One popular library for doing so is called
"CUDA", which is a parallel computing platform and programming model developed by NVIDIA.

To use the GPU with OpenCV, install the opencv-python-headless package,
which includes the CUDA support. Then modify the code to use the cv2.cuda
module instead of the regular cv2 module for GPU-accelerated computations.
However, keep in mind that not all operations in OpenCV are supported by the GPU,
so check the documentation to see which functions are GPU-accelerated.

CUDA homepage:
https://developer.nvidia.com/cuda-zone

link to the CUDA documentation:
https://docs.nvidia.com/cuda/
'''

import cv2
import threading
import queue
import time
# import RPi.GPIO as GPIO
# Load the pre-trained Haar cascade for human detection
human_cascade = cv2.CascadeClassifier('cascades/haarcascade_fullbody.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# # Initialize the servo motors
# servo_a_pin = 27
# servo_b_pin = 22
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(servo_a_pin, GPIO.OUT)
# GPIO.setup(servo_b_pin, GPIO.OUT)
# servo_a = GPIO.PWM(servo_a_pin, 50)  # 50 Hz PWM frequency
# servo_b = GPIO.PWM(servo_b_pin, 50)  # 50 Hz PWM frequency
# servo_a.start(0)  # Initialize the servos to their default position
# servo_b.start(0)

# Define the worker function for human detection and tracking
def worker(input_queue, output_queue):
    while True:
        # Get a frame from the input queue
        frame = input_queue.get()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect humans in the grayscale frame
        humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Put the detected humans into the output queue
        output_queue.put(humans)

        # Mark the input queue item as done
        input_queue.task_done()

# Create a queue for input frames and output humans
input_queue = queue.Queue(maxsize=10)
output_queue = queue.Queue(maxsize=10)

# Create worker threads for human detection and tracking
num_workers = 4
for i in range(num_workers):
    t = threading.Thread(target=worker, args=(input_queue, output_queue))
    t.daemon = True
    t.start()

# Initialize the variables for the low-pass filter
alpha = 0.3
prev_center_x = None
prev_center_y = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Put the frame into the input queue
    input_queue.put(frame)

    # Check if the output queue has any detected humans
    if not output_queue.empty():
        # Get the detected humans from the output queue
        humans = []
        while not output_queue.empty():
            humans.extend(output_queue.get())

        # Draw rectangles around the detected humans
        for (x, y, w, h) in humans:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Move the camera to track the detected human
            center_x = x + w/2
            center_y = y + h/2

            # Apply a low-pass filter to smooth out the camera movement
            if prev_center_x is not None and prev_center_y is not None:
                center_x = alpha * center_x + (1 - alpha) * prev_center_x
                center_y = alpha * center_y + (1 - alpha) * prev_center_y

            prev_center_x = center_x
            prev_center_y = center_y

            # Map the x and y coordinates to servo angles
            angle_a = center_y * (180/480)  # Assumes a 480 pixel high image
            angle_b = center_x * (180/640)  # Assumes a 640 pixel wide image

            # # Set the servo angles
            # servo_a.ChangeDutyCycle(2 + (angle_a/18))
            # servo_b.ChangeDutyCycle(2 + (angle_b/18))
            time.sleep(0.05)  # Allow the servos to reach their target position

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy the window
cap.release()
cv2.destroyAllWindows()