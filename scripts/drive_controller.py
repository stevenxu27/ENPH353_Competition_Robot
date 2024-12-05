#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
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

def spawn_position(pos_num):
    if pos_num == 1:
        position = [0.5, 0, 0.5, 0, 0, 1, 1]
    elif pos_num == 2:
        position = []
    else:
        position = [-4, -2.3, 0.5, 0, 0, 0, 0]

    msg = ModelState()
    msg.model_name = 'B1'

    msg.pose.position.x = position[0]
    msg.pose.position.y = position[1]
    msg.pose.position.z = position[2]
    msg.pose.orientation.x = position[3]
    msg.pose.orientation.y = position[4]
    msg.pose.orientation.z = position[5]
    msg.pose.orientation.w = position[6]

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( msg )

    except rospy.ServiceException:
        print ("Service call failed")


def main():
    rospy.init_node('drive_controller')
    time.sleep(1)

    # # Move to first board
    # move_command(3, 0, 0.35, 3)

    # # Move to second board
    # move_command(10, 0, 0.8, 0.5)
    # move_command(0, -1.5, 0.7, 3)
    # move_command(0, 1.5, 0.7, 0.5)

    # # Move to third board
    # move_command(5, 0, 0.75, 0.5)
    # move_command(0, -2, 1.8, 0.5)
    # move_command(3, 0, 0.5, 0.5)
    # move_command(0, -2, 0.9, 0.5)
    # move_command(3, 0, 0.7, 0.5)
    # move_command(0, 2, 0.8, 3)

    spawn_position(3)
    time.sleep(0.5)
    move_command(3, 0, 3.2, 0.5)
    move_command(0, 1.5, 2.45, 0.5)
    move_command(3, 0, 2.5, 0.5)
    move_command(0, 1.5, 2.45, 0.5)
    move_command(3, 0, 1.8, 0.5)
    move_command(0, 1.5, 2.45, 0.5)
    move_command(3, 0, 1.8, 0)
    move_command(0, 1.5, 2.45, 0.5)
    move_command(3, 0, 0.9, 0.5)
    move_command(0, 1.5, 1.1, 0.5)

if __name__ == "__main__":
    main()