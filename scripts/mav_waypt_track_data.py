#! /usr/bin/env python3

"""
Code to compute reference trajectories that are based on tracking a bunch of random waypoints
"""

import rospy
import numpy as np
import random
import time
from std_srvs.srv import Empty, EmptyResponse


from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, Trigger
from kr_tracker_msgs.msg import TrajectoryTrackerAction, TrajectoryTrackerGoal

from layered_ref_control.mav_layer_interface import KrMavInterface


PI = np.pi


def generate_waypoints(segments):
    """
    Function to generate a list of waypoints given the number of waypoints as an argument
    :return:
    """
    # sample = list()

    # goals = np.random.randn(3, segments+1)
    # return goals.T
    # points = np.linspace(0, 1, segments + 1)
    # for i in range(segments + 1):
    #    sample.append(goals[:, 0] + (goals[:, 1] - goals[:, 0]) * points[i])
    #return np.vstack(sample)
    goal = TrajectoryTrackerGoal()
    wp = Pose()
    wp.position.x = 0.0
    wp.position.y = 0.0
    wp.position.z = 2.0 + np.random.randn(1)
    goal.waypoints.append(wp)

    wp1 = Pose()
    wp1.position.x = 0.0
    wp1.position.y = 1.0 + np.random.randn(1)
    wp1.position.z = 1.5 + np.random.randn(1)
    goal.waypoints.append(wp1)

    wp2 = Pose()
    wp2.position.x = 2.0 + np.random.randn(1)
    wp2.position.y = 2.0 + np.random.randn(1)
    wp2.position.z = 1.0 + np.random.randn(1)
    goal.waypoints.append(wp2)

    wp3 = Pose()
    wp3.position.x = 3.0 + np.random.randn(1)
    wp3.position.y = 0.0 + np.abs(np.random.randn(1))
    wp3.position.z = 0.5 + np.abs(np.random.randn(1))
    goal.waypoints.append(wp3)
    return goal


def main():
    rospy.init_node('mav_traj_data', anonymous=True)

    # Creating MAV objects
    mav_namespace = 'dragonfly'

    mav_obj = KrMavInterface(mav_namespace, "1")

    # Motor On / Take Off

    mav_obj.motors_on()
    mav_obj.take_off()

    rospy.sleep(1)

    for mav_id in range(100):
        # mav_id = 0


        # Send waypoint blocking
        mav_obj.send_wp_block(0.0, 0.0, 1.5, 0.0, 1.0, 0.5, False)  # x, y, z, yaw, vel, acc, relative

        segments = 3
        goal = generate_waypoints(segments=segments+1)
        """goal = TrajectoryTrackerGoal()
        for i in range(segments+1):
            wp = Pose()
            wp.position.x = waypt[i, 0]
            wp.position.y = waypt[i, 1]
            wp.position.z = waypt[i, 2]
            goal.waypoints.append(wp)"""

        mav_obj.traj_tracker_client.send_goal(goal)
        success = mav_obj.transition_service_call('TrajectoryTracker')

        if not success:
            rospy.logwarn("Failed to transition to trajectory tracker (is there an active goal?)")

        rospy.logwarn("Waiting for traj to run")
        mav_obj.traj_tracker_client.wait_for_result()



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    # finally:
    #    rosservice call /rosbag_record_service/stop
