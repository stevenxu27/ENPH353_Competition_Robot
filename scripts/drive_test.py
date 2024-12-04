#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model_path = "/home/fizzer/ros_ws/src/controller/models/drive_model.h5"
model = load_model(model_path)

bridge = CvBridge()
move = Twist()

vel_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)

def image_callback(msg):
    # Convert the ROS Image message to OpenCV format
    image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Resize or preprocess the image to match the model's input size
    resized_image = cv2.resize(image, (224, 224))  # Ensure it matches the model's input
    resized_image = resized_image[56 :, :, :] # Import 3/4 to model
    normalized_image = resized_image / 255.0  # Normalize pixel values to [0, 1]
    
    # Expand dimensions to match model input shape
    input_data = np.expand_dims(normalized_image, axis=0)

    # Compute linear and angular velocities
    prediction = model.predict(input_data)
    linear_vel, angular_vel = prediction[0]

    move.linear.x = linear_vel
    move.angular.z = angular_vel

    vel_pub.publish(move)

def main():
    rospy.init_node('drive_test')

    rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, image_callback)

    rate = rospy.Rate(30)

    rospy.spin()

if __name__ == "__main__":
    main()
    