import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def make_robot(pkg_ioc, namespace, arm_prefix, xyz, controllers_yaml, hold_yaml, arm_yaml):
    xacro_file = os.path.join(pkg_ioc, "urdf", "fr3_3dof.urdf.xacro")
    robot_description_config = xacro.process_file(
        xacro_file,
        mappings={
            "arm_id": "fr3",
            "arm_prefix": arm_prefix,
            "xyz": xyz,
            "controllers_file": os.path.join(pkg_ioc, "config", controllers_yaml),
            "ros2_control": "true",
            "robot_param_node": f"{namespace}_robot_state_publisher",
        }
    )
    robot_description = {"robot_description": robot_description_config.toxml()}
    sim_time_param = {"use_sim_time": True}

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name=f"{namespace}_robot_state_publisher",
        output="screen",
        parameters=[sim_time_param, robot_description],
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

    # Use controller_manager spawner instead of ros2 CLI processes
    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "-c", f"/{namespace}/controller_manager",
        ],
        output="screen",
    )

    hold_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "fixed_joints_hold",
            "-c", f"/{namespace}/controller_manager",
            "--param-file", os.path.join(pkg_ioc, "config", hold_yaml),
        ],
        output="screen",
    )

    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_3dof_cmd",
            "-c", f"/{namespace}/controller_manager",
            "--param-file", os.path.join(pkg_ioc, "config", arm_yaml),
        ],
        output="screen",
    )

    return rsp, spawn, jsb_spawner, hold_spawner, arm_spawner


def generate_launch_description():
    pkg_ioc = get_package_share_directory("iocfranka_3dof")
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

    leader_rsp, leader_spawn, leader_jsb, leader_hold, leader_arm = make_robot(
        pkg_ioc, "leader", "leader_", "-0.50 0 0",
        "iocfranka_leader_controllers.yaml",
        "leader_hold.yaml",
        "leader_arm3.yaml",
    )
    follower_rsp, follower_spawn, follower_jsb, follower_hold, follower_arm = make_robot(
        pkg_ioc, "follower", "follower_", "0.50 0 0",
        "iocfranka_follower_controllers.yaml",
        "follower_hold.yaml",
        "follower_arm3.yaml",
    )



    return LaunchDescription([
        gazebo,
        rviz,

        leader_rsp,
        follower_rsp,
        leader_spawn,
        follower_spawn,

        TimerAction(period=8.0, actions=[leader_jsb]),
        TimerAction(period=10.0, actions=[leader_hold]),
        TimerAction(period=12.0, actions=[leader_arm]),

        TimerAction(period=14.0, actions=[follower_jsb]),
        TimerAction(period=16.0, actions=[follower_hold]),
        TimerAction(period=18.0, actions=[follower_arm]),

    ])
