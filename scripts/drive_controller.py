#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import time
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding

move = Twist()
vel_pub = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)

 
def spawn_position(position):

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

    position = [0.55, -0.1, 0.5, 0.0, 0.0, 1.0, 1.0]  # Spawn at origin with no rotation

    spawn_position(position)  # Spawn at origin with no rotation)

    time.sleep(1)

    move_command(-3, 0, 0.85, 2)
    move_command(0, -1, 1, 4)
    move_command(0, 1, 1, 2)

    time.sleep(1)

    move_command(3, 0, 2.75, 1)
    move_command(0, -3, 1.8, 3)
    move_command(0, -2, 0.5, 1)
    # move_command(1, 0, 1.7, 1)

    time.sleep(1)

    # move_command(0, -2, 2, 1)
    # move_command(2, 1, 1, 1)
    # move_command(2, -1, 1, 1)
    # move_command(2, 0.5, 1, 1)
    # move_command(2, -0.5, 1, 1)
    # move_command(2, 0.5, 1, 1)
    # move_command(2, -0.5, 1, 1)


if __name__ == "__main__":
    main()