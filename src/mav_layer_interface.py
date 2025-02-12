#! /usr/bin/env python3
from __future__ import print_function

import rospy
import actionlib
import numpy as np
import rosbag

import tf
from geometry_msgs.msg import Twist, Pose, Point, Vector3
from kr_mav_msgs.msg import PositionCommand
from nav_msgs.msg import Odometry, Path
from kr_tracker_msgs.msg import LineTrackerAction, LineTrackerGoal, LissajousTrackerAction, LissajousTrackerGoal, TrajectoryTrackerAction, TrajectoryTrackerGoal

from kr_tracker_msgs.srv import Transition
from std_srvs.srv import Trigger, SetBool
from kr_mav_manager.srv import Vec4

from geometry_msgs.msg import Point, PoseArray
from visualization_msgs.msg import MarkerArray, Marker

class KrMavInterface(object):

  def __init__(self, mav_namespace='dragonfly', mav_name="0"):
    self.mav_namespace = mav_namespace
    #self.id = id

    #self.mav_name = self.mav_namespace + str(self.id)
    if mav_name == 1:
      self.mav_name = self.mav_namespace + str(mav_name)
    else:
      self.mav_name = self.mav_namespace + str(mav_name)

    self.pub_waypoint_marker = rospy.Publisher('/' + self.mav_name + '/waypoint_marker', Marker, queue_size=10)

    # Create a ROS publisher for the Path message

    self.path_publisher = rospy.Publisher('/' + self.mav_name + '/ref_traj', Marker, queue_size=10)

    self.path_publisher_replan = rospy.Publisher('/' + self.mav_name + '/replan', Marker, queue_size=10)

    self.pub_vel = rospy.Publisher('/' + self.mav_name + '/cmd_vel', Twist, queue_size=10)
    self.pub_pos_cmd = rospy.Publisher('/' + self.mav_name + '/position_cmd', PositionCommand, queue_size=10)

    self.odom = Odometry()

    self.sub_odom = rospy.Subscriber('/' + self.mav_name + '/odom', Odometry, self.update_odom, queue_size=10)

    self.line_tracker_client = actionlib.SimpleActionClient(self.mav_name +
                                                            '/trackers_manager/line_tracker_min_jerk/LineTracker',
                                                            LineTrackerAction)
    self.lissajous_tracker_client = actionlib.SimpleActionClient(self.mav_name +
                                                            '/trackers_manager/lissajous_tracker/LissajousTracker',
                                                            LissajousTrackerAction)
    self.traj_tracker_client = actionlib.SimpleActionClient(self.mav_name +
                                                            '/trackers_manager/trajectory_tracker/TrajectoryTracker',
                                                            TrajectoryTrackerAction)

    self.traj_tracker_status = ""


  def update_odom(self, msg):
    self.odom = msg


  def get_odom(self):
    return self.odom


  def motors_on(self):
    try:
      motors = rospy.ServiceProxy('/' + self.mav_name + '/mav_services/motors', SetBool)
      resp = motors(True)
      rospy.loginfo(resp)
    except rospy.ServiceException as e:
      rospy.logwarn("Service call failed: %s"%e)
      return 'aborted'


  def motors_off(self):
    try:
      motors = rospy.ServiceProxy('/' + self.mav_name + '/mav_services/motors', SetBool)
      resp = motors(False)
      rospy.loginfo(resp)
    except rospy.ServiceException as e:
      rospy.logwarn("Service call failed: %s"%e)
      return 'aborted'


  def take_off(self):
    try:
      takeoff = rospy.ServiceProxy('/' + self.mav_name + '/mav_services/takeoff', Trigger)
      resp = takeoff()
      rospy.loginfo(resp)
    except rospy.ServiceException as e:
      rospy.logwarn("Service call failed: %s"%e)
      return 'aborted'


  def hover(self):
    rospy.logwarn("Transition to hover")
    self.traj_tracker_client.cancel_all_goals()
    rospy.wait_for_service('/' + self.mav_name  + '/mav_services/hover')
    try:
      srv = rospy.ServiceProxy('/' + self.mav_name + '/mav_services/hover', Trigger)
      resp = srv()
      rospy.loginfo(resp)
    except rospy.ServiceException as e:
      rospy.logwarn("Service call failed: %s" % e)
      return False


  def land(self):
    rospy.logwarn("Transition to land")
    self.traj_tracker_client.cancel_all_goals()
    rospy.wait_for_service('/' + self.mav_name + '/mav_services/land')
    try:
      srv = rospy.ServiceProxy('/' + self.mav_name + '/mav_services/land', Trigger)
      resp = srv()
      rospy.loginfo(resp)
    except rospy.ServiceException as e:
      rospy.logwarn("Service call failed: %s" % e)
      return False


  def set_vel(self, vx=0.0, vy=0.0, vz=0.0, ax=0.0, ay=0.0, az=0.0):
    command = Twist()
    command.linear.x = vx
    command.linear.y = vy
    command.linear.z = vz
    command.angular.x = ax
    command.angular.y = ay
    command.angular.z = az

    self.pub_vel.publish(command)


  def send_wp(self, x, y, z, yaw):
    try:
      rospy.wait_for_service('/' + self.mav_name + '/mav_services/goTo')
      srv = rospy.ServiceProxy('/' + self.mav_name +'/mav_services/goTo', Vec4)
      resp = srv([x, y, z, yaw])
      rospy.loginfo(resp)
    except rospy.ServiceException as e:
      rospy.logwarn("Service call failed: %s" % e)
      return False


  def send_wp_block(self, x, y, z, yaw, vel=1.0, acc=0.5, relative=False):
    goal = LineTrackerGoal()
    goal.x = x
    goal.y = y
    goal.z = z
    goal.yaw = yaw
    goal.v_des = 1.0
    goal.a_des = 0.5
    goal.relative = False

    self.line_tracker_client.send_goal(goal)
    self.transition_service_call('LineTrackerMinJerk')
    rospy.logwarn("Waiting for Line Tracker to complete")
    self.line_tracker_client.wait_for_result()

  def publish_waypoints(self, waypoints, r, g, b, a):
    # Create a Marker message
    waypoint_marker = Marker()
    waypoint_marker.header.stamp = rospy.Time.now()
    waypoint_marker.header.frame_id = 'simulator'
    waypoint_marker.ns = 'waypoints'
    waypoint_marker.id = 1
    waypoint_marker.type = Marker.POINTS
    waypoint_marker.action = Marker.ADD

    try:
      # Create a marker for each waypoint
      for i in range(waypoints.shape[1]):
        point = Point()
        point.x = waypoints[0, i]
        point.y = waypoints[1, i]
        point.z = waypoints[2, i]
        waypoint_marker.points.append(point)

      waypoint_marker.scale.x = 0.1
      waypoint_marker.scale.y = 0.1
      waypoint_marker.scale.z = 0.1
      waypoint_marker.color.r = r
      waypoint_marker.color.g = g
      waypoint_marker.color.b = b
      waypoint_marker.color.a = a

      # Publish the marker
      self.pub_waypoint_marker.publish(waypoint_marker)

    except TypeError as e:
      rospy.logerr(f"Error creating waypoint marker: {str(e)}")

  def publish_ref_traj(self, ref_traj):
    path = Marker()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = 'simulator'
    path.ns = 'ref_traj'
    path.id = 2
    path.type = Marker.LINE_STRIP
    path.action = Marker.ADD

    # LINE_STRIP markers use only the x component of scale, for the line width
    path.scale.x = 0.02
    # path.scale.y = 0.05
    # path.scale.z = 0.05
    path.color.r = 0.0
    path.color.g = 0.0
    path.color.b = 1.0
    path.color.a = 1.0

    try:
      # Create a marker for each waypoint
      for i in range(ref_traj.shape[1]):
        point = Point()
        point.x = ref_traj[0, i]
        point.y = ref_traj[1, i]
        point.z = ref_traj[2, i]
        # rospy.logwarn(f"Waypoint {i}: {point.x}, {point.y}, {point.z}")
        path.points.append(point)

      self.path_publisher.publish(path)

    except TypeError as e:
      rospy.logerr(f"Error creating ref traj marker: {str(e)}")


  def publish_replan(self, ref_traj):
    path = Marker()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = 'simulator'
    path.ns = 'replan'
    path.id = 3
    path.type = Marker.LINE_STRIP
    path.action = Marker.ADD

    # LINE_STRIP markers use only the x component of scale, for the line width
    path.scale.x = 0.02
    # path.scale.y = 0.05
    # path.scale.z = 0.05
    path.color.r = 0.0
    path.color.g = 1.0
    path.color.b = 0.0
    path.color.a = 1.0

    try:
      # Create a marker for each waypoint
      for i in range(ref_traj.shape[1]):
        point = Point()
        point.x = ref_traj[0, i]
        point.y = ref_traj[1, i]
        point.z = ref_traj[2, i]
        # rospy.logwarn(f"Waypoint {i}: {point.x}, {point.y}, {point.z}")
        path.points.append(point)

      self.path_publisher.publish(path)

    except TypeError as e:
      rospy.logerr(f"Error creating ref traj marker: {str(e)}")

  def transition_service_call(self, tracker_name):
    rospy.loginfo('waiting for transition service for ' + tracker_name)
    rospy.wait_for_service('/' + self.mav_name +'/trackers_manager/transition')
    try:
      tt = rospy.ServiceProxy('/' + self.mav_name +'/trackers_manager/transition', Transition)
      resp = tt('kr_trackers/' + tracker_name)
      rospy.loginfo(resp)
      if resp.success == False or "already active" in resp.message:
        return False
      return True
    except rospy.ServiceException as e:
      rospy.logwarn("Service call failed: %s" % e)
      return False


  def publish_pos_cmd(self, pos0, pos1, pos2, vel0, vel1, vel2, acc0, acc1, acc2, jerk0, jerk1, jerk2, yaw, yaw_dot):
    command = PositionCommand()
    command.header.stamp = rospy.Time.now()
    command.position.x = pos0
    command.position.y = pos1
    command.position.z = pos2
    command.velocity.x = vel0
    command.velocity.y = vel1
    command.velocity.z = vel2
    command.acceleration.x = acc0
    command.acceleration.y = acc1
    command.acceleration.z = acc2
    command.jerk.x = jerk0
    command.jerk.y = jerk1
    command.jerk.z = jerk2
    command.yaw = yaw
    command.yaw_dot = yaw_dot
    self.pub_pos_cmd.publish(command)
