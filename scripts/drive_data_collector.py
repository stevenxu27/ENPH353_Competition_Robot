#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import csv

# Initialize CvBridge
bridge = CvBridge()

# Initialize data array
data = []

# Initialize callback variables
latest_image = None
latest_velocity = None

# Callback for image data
def image_callback(msg):
    global latest_image
    try:
        # Convert the ROS Image message to OpenCV format
        latest_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logerr(f"Failed to convert image: {e}")
        latest_image = None

# Callback for velocity data
def velocity_callback(msg):
    global latest_velocity
    latest_velocity = msg


def main():
    global latest_image, latest_velocity, data

    # Initialize the ROS node
    rospy.init_node('data_collector')

    # Subscribe to the image and velocity topics
    rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, image_callback)
    rospy.Subscriber('/B1/cmd_vel', Twist, velocity_callback)

    rospy.loginfo("Data collection node started. Press Ctrl+C to stop.")

    rate = rospy.Rate(2)

    # Keep the node running
    try:
        while not rospy.is_shutdown():
            if latest_image is not None and latest_velocity is not None:
                # Save the image and velocity data
                image_array = latest_image.copy()
                linear_vel = latest_velocity.linear.x
                angular_vel = latest_velocity.angular.z

                data.append((image_array, linear_vel, angular_vel))

                # Optionally, display the image (useful for debugging)
                # cv2.imshow('Current Image', image_array)
                # cv2.waitKey(1)

            # Sleep to maintain the rate
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        # Save data to disk when stopping
        rospy.loginfo("Saving data...")
        save_data()
        rospy.loginfo("Data saved successfully.")

def save_data():
    global data

    # Save images and velocity data
    for idx, (image, linear_vel, angular_vel) in enumerate(data):
        # Save the image
        cv2.imwrite(f'/home/fizzer/ros_ws/src/controller/drive_data_output/image_{idx}.jpg', image)

        # Append to a CSV file
        with open('/home/fizzer/ros_ws/src/controller/drive_data_output/velocity_data.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([f'image_{idx}.jpg', linear_vel, angular_vel])

if __name__ == '__main__':
    main()