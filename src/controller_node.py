#!/usr/bin/env python3

import rospy
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def main():
    # Initialize the ROS node
    rospy.init_node('controller_node', anonymous=True)
    # sleep after initialization
    time.sleep(1)

    # Create a publisher to send velocity commands and score tracker
    image_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
    score_pub = rospy.Publisher('/score_tracker', String)

    rospy.sleep(1)

    # Create a rate object to control the publishing rate (30 Hz)
    rate = rospy.Rate(30)  # 30 Hz loop rate

    team_name = "The McDoubles"
    password = "password"

    start_message = f"{team_name},{password},0, Starting Simulation: Moving Forward"
    rospy.loginfo(f"Publishing start message: {start_message}")
    score_pub.publish(start_message)

    move = Twist()
    move.linear.x = 0.25
    move.angular.z = 0.1
    image_pub.publish(move)
    rate.sleep()

    rospy.sleep(20)

    move.linear.x = 0
    move.angular.z = 0
    image_pub.publish(move)

    stop_message = f"{team_name},{password},-1,Finishing Simulation: Stop"
    rospy.loginfo(f"Publishing stop message: {stop_message}")
    score_pub.publish(stop_message)
    rate.sleep()

    rospy.sleep(1)

    # while not rospy.is_shutdown():
    #     # Publish the move command
    #     image_pub.publish(move)

    #     # Sleep to maintain the rate
    #     rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down.")

