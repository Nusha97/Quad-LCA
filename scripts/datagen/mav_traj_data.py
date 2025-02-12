#! /usr/bin/env python3
import rospy
import numpy as np
import random
import time
from rosbag_record_service import *
from std_srvs.srv import Empty, EmptyResponse


from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool, Trigger
from kr_tracker_msgs.msg import TrajectoryTrackerAction, TrajectoryTrackerGoal, CircleTrackerAction, CircleTrackerGoal, LissajousTrackerGoal, LissajousTrackerAction

from layered_ref_control.mav_layer_interface import KrMavInterface


PI = np.pi


def generate_waypoints(segments):
    """
    Function to generate a list of waypoints given the number of waypoints as an argument
    :return:
    """
    waypoints = np.random.randn(p, segments + 1)
    waypoints[:, 0] = 0
    waypoints[:, -1] = 0
    ts = np.linspace(0, 1, segments + 1) * 3
    return waypoints


def generate_lissajous_traj(s, x_num_periods, y_num_periods, z_num_periods, yaw_num_periods, period, x_amp, y_amp, z_amp, yaw_amp,
                  ramp_time, num_cycles):
    """
    Function to generate Lissajous trajectory
    :return:
    """
    x = lambda a: x_amp * (1 - np.cos(2 * PI * x_num_periods * a / period))
    y = lambda a: y_amp * np.sin(2 * PI * y_num_periods * a / period)
    z = lambda a: z_amp * np.sin(2 * PI * z_num_periods * a / period)
    yaw = lambda a: yaw_amp * np.sin(2 * PI * yaw_num_periods * a / period)
    return [x(s), y(s), z(s), yaw(s)]


### Write script here to check the time at which rosbag starts saving first trajectory
###

def main():
    rospy.init_node('mav_traj_data', anonymous=True)

    # Creating MAV objects
    mav_namespace = 'dragonfly'

    for mav_id in range(100):
        # mav_id = 1

        """try:
            rospy.wait_for_service('/rosbag_record_service/start')
            # start_service = rospy.ServiceProxy('/rosbag_record_service/start', EmptyResponse)
            print("What's happening? Did it start?")
            start_service()
            print("What's happening?")
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)"""
        # if mav_id >= 1:
        #    continue
        # Motor On / Take Off

        mav_obj = KrMavInterface(mav_namespace, mav_id + 1)

        mav_obj.motors_on()
        mav_obj.take_off()

        rospy.sleep(1)

        # Send waypoint blocking
        mav_obj.send_wp_block(0.0, 0.0, 1.5, 0.0, 1.0, 0.5, False)  # x, y, z, yaw, vel, acc, relative

        """traj_list = generate_traj(s=np.linspace(0, 3, 10), x_num_periods=12, y_num_periods=12, z_num_periods=16,
                                  yaw_num_periods=5.0, period=60.0, x_amp=1.25 * random.random(),
                                  y_amp=1.25 * random.random(),
                                  z_amp=0.75 * random.random(), yaw_amp=3.1415, ramp_time=2.0, num_cycles=1.0)

        lissajous_traj = traj_list.copy()
        goal = TrajectoryTrackerGoal()
        for i in range(10):
            wp = Pose()
            wp.position.x = lissajous_traj[0][i]
            wp.position.y = lissajous_traj[1][i]
            wp.position.z = lissajous_traj[2][i]
            goal.waypoints.append(wp)

        mav_obj.traj_tracker_client.send_goal(goal)
        success = mav_obj.transition_service_call('TrajectoryTracker')"""
        goal = LissajousTrackerGoal()
        # for i in range(10):
        goal.x_amp = 1.25 * random.random()
        goal.y_amp = 1.25 * random.random()
        goal.z_amp = 0.75 * random.random()
        goal.yaw_amp = 3.1414 * random.random()
        goal.x_num_periods = 1
        goal.y_num_periods = 1
        goal.z_num_periods = 1
        goal.yaw_num_periods = 1
        goal.period = 3
        goal.num_cycles = 1
        goal.ramp_time = 2

        mav_obj.lissajous_tracker_client.send_goal(goal)
        success = mav_obj.transition_service_call('LissajousTracker')


        if not success:
            rospy.logwarn("Failed to transition to trajectory tracker (is there an active goal?)")

        rospy.logwarn("Waiting for traj to run")
        # mav_obj.traj_tracker_client.wait_for_result()
        mav_obj.lissajous_tracker_client.wait_for_result()

        # do trajectory tracking for a bunch of random waypoints

        # rosservice call /rosbag_record_service/pause_resume
        del mav_obj
        """try:
            rospy.wait_for_service('/rosbag_record_service/start')
            # pause_resume_service = rospy.ServiceProxy('/rosbag_record_service/start', EmptyResponse)
            pause_resume_service()
        except rospy.ServiceException as e:
            rospy.logwarn("Service call failed: %s" % e)"""



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    # finally:
    #    rosservice call /rosbag_record_service/stop
