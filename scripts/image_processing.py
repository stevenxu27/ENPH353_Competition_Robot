#!/usr/bin/env python3

import rospy
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def image_callback(msg):
    bridge = CvBridge()
    try:
        # Convert ROS Image message to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("Received Homography", cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("Shutting down subscriber.")
    except Exception as e:
        rospy.logerr(f"Error converting image: {e}")


def main():

    rospy.init_node('homography_viewer', anonymous=True)

    time.sleep(1)

    rospy.Subscriber('/homography_result', Image, image_callback)

    time.sleep(1)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down.")