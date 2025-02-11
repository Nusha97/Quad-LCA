cmake_minimum_required(VERSION 3.0.2)
project(layered_ref_control)

find_package(catkin REQUIRED roscpp rosbag message_filters cv_bridge image_transport kr_mav_msgs)

catkin_python_setup()

catkin_package(
CATKIN_DEPENDS roscpp rosbag message_filters cv_bridge image_transport kr_mav_msgs)

include_directories(${catkin_INCLUDE_DIRS})

add_executable(sync_msg src/sync_msg.cpp)
target_link_libraries(sync_msg
   ${catkin_LIBRARIES} ${Boost_LIBRARIES}
)

install(PROGRAMS src/layered_ref_control/mav_layer_interface.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)