#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model

model_path = "/home/fizzer/ros_ws/src/controller/models/no_spaces_model.h5"
model = load_model(model_path)

# Define the image size you want to slice
image_height, image_width = 400, 600
slice_height, slice_width = 100, 50  # Example: 40x40 slices for each letter


classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '  # String of values
class_to_index = {char: idx for idx, char in enumerate(classes)}  # Map char -> index
index_to_class = {idx: char for char, idx in class_to_index.items()}  # Map index -> char

def preprocess_image(img):
    # Resize to the size the model expects (100, 50, 3)
    # img_resized = cv2.resize(img, (slice_width, slice_height))
    # img_normalized = img_resized / 255.0  # Normalize to [0, 1]
    img_expanded = np.expand_dims(img, axis=0)  # Add batch dimension
    return img_expanded

def predict_letter(roi):
    
    # Predict using the model
    roi_preprocessed = preprocess_image(roi)
    prediction = model.predict(roi_preprocessed)
    
    # Get the predicted class (for classification models)
    predicted_class = np.argmax(prediction)
    
    return predicted_class

def slice_image(image):
    # Slice the image into 4 sections (adjust these coordinates as needed)
    first_character = image[250:350, 25:75]
    second_character = image[250:350, 75:125]
    third_character = image[250:350, 120:170]
    fourth_character = image[250:350, 170:220]
    fifth_character = image[250:350, 215:265]
    sixth_character = image[250:350, 265:315]
    seventh_character = image[250:350, 320:370]
    eigth_character = image[250:350, 375:425]
    ninth_character = image[250:350, 425:475]

    return first_character, second_character, third_character, fourth_character, fifth_character, sixth_character, seventh_character, eigth_character, ninth_character

def image_callback(msg):
    bridge = CvBridge()
    try:
        # Convert ROS Image message to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        rospy.loginfo(f"OpenCV image shape: {cv_image.shape}")

        if cv_image is None or cv_image.size == 0:
            rospy.logerr("Received an empty or invalid image from the camera feed!")
            return
        
        # cv2.imshow("Received Homography", cv_image[250:350, 25:75])

        first, second, third, fourth, fifth, sixth, seventh, eighth, ninth = slice_image(cv_image)

        message = ''
        for i, character in enumerate([first, second, third, fourth, fifth, sixth]):
            rospy.loginfo(f"OpenCV image shape: {character.shape}")
            cv2.imshow('Test character', character)
            predicted_class = predict_letter(character)
            message += str(index_to_class.get(predicted_class))
            rospy.loginfo(f"Predicted class for character {i+1}: {predicted_class}")
        rospy.loginfo(f"Predicted message: {message}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("Shutting down subscriber.")
    except Exception as e:
        rospy.logerr(f"Error converting image: {e}")


def main():

    rospy.init_node('image_processing', anonymous=True)

    time.sleep(1)

    rospy.Subscriber('/homography_result', Image, image_callback)

    time.sleep(1)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down.")