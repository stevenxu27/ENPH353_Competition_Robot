#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class PIDDriveNode:
    def __init__(self):
        rospy.init_node('pid_drive', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.camera_callback)
        self.cmd_vel_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=10)

        # PID control parameters
        self.kp = 0.2
        self.ki = 0.01
        self.kd = 0.02
        self.prev_error = 0
        self.integral = 0
        self.last_time = rospy.get_time()

        # Fallback velocity
        self.base_speed = 0.3  # Reduced to slow down near edges
        self.last_steering_angle = 0.0
        self.max_angular_velocity = 0.5  # Limit steering adjustments

        # Edge detection parameters
        self.edge_threshold = 50  # Threshold for edge detection
        self.sobel_threshold = 50  # Gradient threshold for edge detection

    def camera_callback(self, data):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Crop to ROI for white line detection
            height, width, _ = cv_image.shape
            roi = cv_image[int(0.6 * height):, :]  # Bottom 40% of the image

            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # Detect the cliff edge using Sobel
            edge_detected = self.detect_edge(gray)

            # Calculate the centroid of the white lines
            moments = cv2.moments(binary)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                error = cx - width // 2
                self.control_car(error, edge_detected)
            else:
                # No lines detected, fallback to avoid edge
                rospy.logwarn("Lines not detected, avoiding edge.")
                self.control_car(0, edge_detected, use_last=True)
        except Exception as e:
            rospy.logerr(f"Camera callback error: {e}")

    def detect_edge(self, gray):
        # Apply Sobel filter to detect edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        mean_gradient = np.mean(sobel)

        # Detect edges based on gradient magnitude
        return mean_gradient < self.sobel_threshold

    def control_car(self, error, edge_detected, use_last=False):
        # If edge is detected, stop or steer away
        if edge_detected:
            rospy.logwarn("Cliff detected! Stopping to avoid falling.")
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return

        # PID control calculations
        current_time = rospy.get_time()
        dt = current_time - self.last_time

        if not use_last:
            self.integral += error * dt
            derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
            steering_angle = self.kp * error + self.ki * self.integral + self.kd * derivative
            self.prev_error = error
        else:
            steering_angle = self.last_steering_angle

        # Cap the steering angle to avoid aggressive turns
        self.last_steering_angle = max(-self.max_angular_velocity, min(steering_angle, self.max_angular_velocity))
        self.last_time = current_time

        # Publish velocity command
        twist = Twist()
        twist.linear.x = self.base_speed
        twist.angular.z = -self.last_steering_angle / 200.0  # Scaled for smoother turns
        self.cmd_vel_pub.publish(twist)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PIDDriveNode()
        node.run()
    except rospy.ROSInterruptException:
        pass