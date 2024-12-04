#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import time

move = Twist()
vel_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)

def move_command(lin_x, ang_z, move_time, wait_time):
    move.linear.x = lin_x
    move.angular.z = ang_z
    vel_pub.publish(move)
    time.sleep(move_time)
    move.linear.x = 0
    move.angular.z = 0
    vel_pub.publish(move)
    if wait_time > 0:
        time.sleep(wait_time)

def main():
    rospy.init_node('drive_controller')
    time.sleep(1)

    # Move to first board
    move_command(3, 0, 0.35, 3)

    # Move to second board
    move_command(10, 0, 0.8, 0.5)
    move_command(0, -1.5, 0.7, 3)
    move_command(0, 1.5, 0.7, 0.5)

    # Move to third board
    move_command(5, 0, 0.75, 0.5)
    move_command(0, -2, 1.8, 0.5)
    move_command(3, 0, 0.5, 0.5)
    move_command(0, -2, 0.9, 0.5)
    move_command(3, 0, 0.7, 0.5)
    move_command(0, 2, 0.8, 3)


if __name__ == "__main__":
    main()