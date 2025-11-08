cmake_minimum_required(VERSION 3.8)
project(snaak_shredded_grasp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetGraspPose.srv"
)

install(PROGRAMS
  scripts/snaak_shredded_grasp_node.py
  scripts/snaak_shredded_grasp_utils.py
  scripts/snaak_shredded_grasp_constants.py
  scripts/granular_grasp_utils.py
  scripts/classical_grasp_utils.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_python_install_package(scripts)
ament_package()