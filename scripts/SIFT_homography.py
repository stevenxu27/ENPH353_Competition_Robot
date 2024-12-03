#!/usr/bin/env python3

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

class Sign_Detection():
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('Sign_Detection', anonymous=True)

        time.sleep(1)

        # Create a CvBridge instance
        self.bridge = CvBridge()

        # Load the template image
        self.template_image = cv2.imread('/home/fizzer/ros_ws/src/controller/images/clue_banner.png')

        self.result_pub = rospy.Publisher('/homography_result', Image, queue_size=10)

        time.sleep(1)

        # Check if template image is loaded
        if self.template_image is None:
            rospy.logerr("Failed to load template image.")
            rospy.signal_shutdown("Template image missing")

        # Convert the template image to grayscale
        self.template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)

        # Subscribe to the camera feed topic
        rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.image_callback)

        time.sleep(1)
        rospy.loginfo("Camera feed processor node started. Waiting for images...")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process the image
            self.process_frame(cv_image)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image message to OpenCV format: {e}")

    def process_frame(self, frame):
        # Detect keypoints and descriptors in both the template and current frame using SIFT
        sift = cv2.SIFT_create()
        kp_template, desc_template = sift.detectAndCompute(self.template_gray, None)
        kp_frame, desc_frame = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)

        # Match descriptors using FLANN
        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), {})
        matches = flann.knnMatch(desc_template, desc_frame, k=2)

        # Filter good matches using Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

        if len(good_matches) >= 4:

            rospy.loginfo("Enough points were detected for good image")
            # Extract matching points
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find the homography matrix
            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Get the corners of the template image
            h, w = self.template_image.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            # Apply homography transformation to the corners
            dst = cv2.perspectiveTransform(pts, matrix)

            # Draw the homography polygon on the frame
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

            # PUBLISH CROPPED IMAGE

            # Draw the homography polygon on the frame (for visualization)
            cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

            # Warp the frame to extract the region inside the homography box
            warp_matrix = cv2.getPerspectiveTransform(dst.reshape(-1, 2), pts.reshape(-1, 2))
            warped_image = cv2.warpPerspective(frame, warp_matrix, (w, h))

            # Publish the cropped region
            try:
                cropped_image_msg = self.bridge.cv2_to_imgmsg(warped_image, encoding="bgr8")
                self.result_pub.publish(cropped_image_msg)
                rospy.loginfo("Published homography ROI")
            except CvBridgeError as e:
                rospy.logerr(f"Failed to publish ROI: {e}")

            # Show the frame with the homography
            cv2.imshow("Homography", warped_image)

        else:
            rospy.logwarn("Not enough good matches found to compute homography")

        # Wait for a key press and close the window on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("Shutting down node.")

    def run(self):
        # Keep the node running
        rospy.spin()

        # Cleanup OpenCV windows on shutdown
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        processor = Sign_Detection()
        processor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Camera feed processor node terminated.")
