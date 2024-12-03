#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import csv
import os

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

        # Resize the image to 128x128
        # latest_image = cv2.resize(original_image, (400, 400))
    except Exception as e:
        rospy.logerr(f"Failed to process image: {e}")
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

    rate = rospy.Rate(10) # Data collection rate

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

    # Define paths
    output_dir = '/home/fizzer/ros_ws/src/controller/train/drive_data_output'
    csv_path = os.path.join(output_dir, 'velocity_data.csv')

    # Find the highest image index in the output folder
    existing_images = [f for f in os.listdir(output_dir) if f.startswith('image_') and f.endswith('.jpg')]
    if existing_images:
        max_index = max(int(f.split('_')[1].split('.')[0]) for f in existing_images)
    else:
        max_index = -1

    # Start saving new data from the next available index
    start_index = max_index + 1

    # Exclude the first 3 and last 3 images
    cutout_frames = 10
    if len(data) > 2 * cutout_frames:
        data = data[cutout_frames:-1 * cutout_frames]  # Slice out the first and last 3 items
    else:
        print("Not enough data to exclude the first and last 3 images.")
        return

    # Save images and velocity data
    for idx, (image, linear_vel, angular_vel) in enumerate(data):
        # Save the image
        image_index = start_index + idx
        image_path = os.path.join(output_dir, f'image_{image_index}.jpg')
        cv2.imwrite(image_path, image)

        # Append to the CSV file
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([f'image_{image_index}.jpg', linear_vel, angular_vel])

    print(f"Saved {len(data)} images and updated CSV file.")

if __name__ == '__main__':
    main()