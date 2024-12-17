# Autonomous Robot Competition

Developing an autonomous robot in ROS Gazebo to navigate through an obstacle course and read clue boards using a trained character recognition model.

## Getting Started

These instructions will set up the environment to deploy the simulation, score_tracker, and controller and image_processing nodes simulaneously. 


## Usage

A few examples of useful commands and/or tasks.

Build Environment:
```

cd ~/ros_ws
catkin_make

```
Source environment:
```

source ~/ros_ws/devel/setup.bash

```

Start the simulated world:
```

cd ~/ros_ws/src/2024_competition/enph353/enph353_utils/scripts
./run_sim.sh -vpg

```
Start score_tracker:
```

cd ~/ros_ws/src/2024_competition/enph353/enph353_utils/scripts
./score_tracker.py

```
To launch image processing and controller node:
```

cd ~/ros_ws/src/controller/launch
roslaunch my_launch.launch

```
These three will need to be launched in 3 separate terminals. The simulation must be launched first, followed by a score tracker and controller.

To view the simulation from the robot's camera feed:
```

rosrun rqt_image_view rqt_image_view

```


### Branches

* main: Most up-to-date implemented drive and image recognition.
* image_processing: Optimizing different models to train the most efficient model for character recognition from a different perspective.
* imitation_learning_drive: Training our imitation learning drive around the course.
* initial_drive_testing: Basic code for time trials.

