#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
from std_msgs.msg import String
from tensorflow.keras.models import load_model

model_path = "/home/fizzer/ros_ws/src/controller/models/working_model.h5"
model = load_model(model_path)

# Define the image size you want to slice
image_height, image_width = 400, 600
slice_height, slice_width = 100, 45  # Example: 40x40 slices for each letter


classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '  # String of values
class_to_index = {char: idx for idx, char in enumerate(classes)}  # Map char -> index
index_to_class = {idx: char for char, idx in class_to_index.items()}  # Map index -> char

clue_map = {
    'SIZE': '1',
    'VICTIM': '2',
    'CRIME': '3',
    'TIME': '4',
    'PLACE': '5',
    'MOTIVE': '6',
    'WEAPON': '7',
    'BANDIT': '8',
}


score_pub = rospy.Publisher('/score_tracker', String)

def preprocess_image(img):
    img_expanded = np.expand_dims(img, axis=0)  # Add batch dimension
    return img_expanded

def predict_letter(roi):

    roi_preprocessed = preprocess_image(roi)
    prediction = model.predict(roi_preprocessed)
    predicted_class = np.argmax(prediction)
    
    return predicted_class

def slice_image(image):
    # Slice the image into 4 sections (adjust these coordinates as needed)
    first_character = image[250:350, 30:75]
    second_character = image[250:350, 75:120]
    third_character = image[250:350, 120:165]
    fourth_character = image[250:350, 165:210]
    fifth_character = image[250:350, 210:255]
    sixth_character = image[250:350, 255:300]
    seventh_character = image[250:350, 300:345]
    eigth_character = image[250:350, 345:390]
    ninth_character = image[250:350, 390:435]
    tenth_character = image[250:350, 435:480]
    eleventh_character = image[250:350, 480:525]
    twelvth_character = image[250:350, 525:570]
    return first_character, second_character, third_character, fourth_character, fifth_character, sixth_character, seventh_character, eigth_character, ninth_character, tenth_character, eleventh_character, twelvth_character

def slice_clue(image):
    first_character = image[30:130, 250:295]
    second_character = image[30:130, 295:340]
    third_character = image[30:130, 340:385]
    fourth_character = image[30:130, 385:430]
    fifth_character = image[30:130, 430:475]
    sixth_character = image[30:130, 475:520]
    return first_character, second_character, third_character, fourth_character, fifth_character, sixth_character

def find_clue(message):

    return clue_map.get(message, '100')

def is_between_0_and_6(s):
    if s.isdigit():  # Check if the string is numeric
        number = int(s)  # Convert to an integer
        return 0 <= number <= 6  # Check if it's in the range
    return False  # Return False if not numeric

def image_callback(msg):

    global counter  # Declare counter as global to modify it
    bridge = CvBridge()
    try:
        # Convert ROS Image message to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        # rospy.loginfo(f"OpenCV image shape: {cv_image.shape}")

        if cv_image is None or cv_image.size == 0:
            rospy.logerr("Received an empty or invalid image from the camera feed!")
            return
        
        # cv2.imshow("Received Homography", cv_image[250:350, 25:75])

        first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelvth = slice_image(cv_image)
        first_character, second_character, third_character, fourth_character, fifth_character, sixth_character = slice_clue(cv_image)

        clue = ''
        for i, character in enumerate([first_character, second_character, third_character, fourth_character, fifth_character, sixth_character]):
            cv2.imshow('Test character', character)
            predicted_class = predict_letter(character)
            clue += str(index_to_class.get(predicted_class))

        message = ''
        for i, character in enumerate([first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, twelvth]):
            # cv2.imshow('Test character', character)
            predicted_class = predict_letter(character)
            message += str(index_to_class.get(predicted_class))

        team_name = "The McDoubles"
        password = "Junior Chicken"

        clue = clue.replace(" ", "")
        
        if (is_between_0_and_6(find_clue(clue))):
            start_message = f"{team_name},{password},{find_clue(clue)}, {message}"
            score_pub.publish(start_message)
        rospy.loginfo(f"Clue: {clue}")
        rospy.loginfo(f"message: {message}")

        rospy.loginfo(f"Predicted message: {message}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("Shutting down subscriber.")
    except Exception as e:
        rospy.logerr(f"Error converting image: {e}")


def main():

    rospy.init_node('image_processing', anonymous=True)

    time.sleep(1)

    team_name = "The McDoubles"
    password = "password"

    start_message = f"{team_name},{password},0, Starting Simulation: Moving Forward"
    rospy.loginfo(f"Publishing start message: {start_message}")
    score_pub.publish(start_message)

    rospy.Subscriber('/homography_result', Image, image_callback)

    time.sleep(1)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down.")