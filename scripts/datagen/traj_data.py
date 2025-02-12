"""
Sample trajectories using SO3 controller
"""


import numpy as np
import rospy
import random
import time

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, Trigger
from kr_tracker_msgs.msg import TrajectoryTrackerAction, TrajectoryTrackerGoal

from kr_python_interface.mav_interface import KrMavInterface

def main():
  rospy.init_node('mav_example', anonymous=True)

  # Creating MAV objects
  mav_namespace = 'dragonfly'
  mav_id = 1
  mav_obj = KrMavInterface('dragonfly', mav_id)

  # Motor On / Take Off
  mav_obj.motors_on()
  mav_obj.take_off()

  rospy.sleep(1)

  # Send waypoint blocking
  mav_obj.send_wp_block(0.0, 0.0, 1.5, 0.0, 1.0, 0.5, False)  # x, y, z, yaw, vel, acc, relative

  # Send multiple waypoints by fitting traj
  goal = TrajectoryTrackerGoal()
  wp = Pose()
  wp.position.x = 0.0
  wp.position.y = 0.0
  wp.position.z = 2.0
  goal.waypoints.append(wp)

  wp1 = Pose()
  wp1.position.x = 0.0
  wp1.position.y = 1.0
  wp1.position.z = 1.5
  goal.waypoints.append(wp1)

  wp2 = Pose()
  wp2.position.x = 2.0
  wp2.position.y = 2.0
  wp2.position.z = 1.0
  goal.waypoints.append(wp2)

  wp3 = Pose()
  wp3.position.x = 3.0
  wp3.position.y = 0.0
  wp3.position.z = 0.5
  goal.waypoints.append(wp3)

  mav_obj.traj_tracker_client.send_goal(goal)
  success = mav_obj.transition_service_call('TrajectoryTracker')
  if not success:
      rospy.logwarn("Failed to transition to trajectory tracker (is there an active goal?)")

  rospy.logwarn("Waiting for traj to run")
  mav_obj.traj_tracker_client.wait_for_result()