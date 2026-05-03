import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def make_robot(pkg_name, namespace, arm_prefix, xyz):
    pkg_share = get_package_share_directory(pkg_name)
    xacro_file = os.path.join(pkg_share, "urdf", "fr3_3dof.urdf.xacro")
    robot_description_config = xacro.process_file(
        xacro_file,
        mappings={
            "arm_id": "fr3",
            "arm_prefix": arm_prefix,
            "xyz": xyz,
            "ros2_control": "false",
            "controllers_file": "",
        }
    )
    robot_description = {"robot_description": robot_description_config.toxml()}

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name=f"{namespace}_robot_state_publisher",
        output="screen",
        parameters=[robot_description],
        remappings=[
            ("/robot_description", f"/{namespace}/robot_description"),
            ("/tf", f"/{namespace}/tf"),
            ("/tf_static", f"/{namespace}/tf_static"),
        ],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", f"/{namespace}/robot_description",
            "-name", namespace,
            "-allow_renaming", "false",
            "-x", xyz.split()[0],
            "-y", xyz.split()[1],
            "-z", xyz.split()[2],
        ],
        output="screen",
    )
    return [rsp, spawn]


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_franka_desc = get_package_share_directory("franka_description")

    os.environ["GZ_SIM_RESOURCE_PATH"] = os.path.dirname(pkg_franka_desc)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": "empty.sdf -r"}.items(),
    )

    rviz_file = os.path.join(pkg_franka_desc, "rviz", "visualize_franka.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["--display-config", rviz_file, "-f", "world"],
        output="screen",
    )

    leader = make_robot("franka_leader_3dof", "leader", "leader_", "-0.50 0 0")
    follower = make_robot("franka_follower_3dof", "follower", "follower_", "0.50 0 0")

    return LaunchDescription([gazebo, rviz] + leader + follower)
