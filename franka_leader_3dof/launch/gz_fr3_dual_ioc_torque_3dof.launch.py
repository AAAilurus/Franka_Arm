import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def make_robot(pkg_name, namespace, arm_prefix, xyz, controllers_yaml):
    pkg_share = get_package_share_directory(pkg_name)
    xacro_file = os.path.join(pkg_share, "urdf", "fr3_3dof.urdf.xacro")
    controllers_file = os.path.join(pkg_share, "config", controllers_yaml)

    robot_description_config = xacro.process_file(
        xacro_file,
        mappings={
            "arm_id": "fr3",
            "arm_prefix": arm_prefix,
            "xyz": xyz,
            "controllers_file": controllers_file,
            "ros2_control": "true",
            "robot_param_node": f"{namespace}_robot_state_publisher",
        }
    )
    robot_description = {"robot_description": robot_description_config.toxml()}

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=namespace,
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
        ],
        output="screen",
    )

    arm_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_3dof_cmd",
            "-c", f"/{namespace}/controller_manager",
        ],
        output="screen",
    )

    return rsp, spawn, jsb_spawner, hold_spawner, arm_spawner


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

    leader_rsp, leader_spawn, leader_jsb, leader_hold, leader_arm = make_robot(
        "franka_leader_3dof", "leader", "leader_", "-0.50 0 0", "franka_leader_controllers.yaml"
    )
    follower_rsp, follower_spawn, follower_jsb, follower_hold, follower_arm = make_robot(
        "franka_follower_3dof", "follower", "follower_", "0.50 0 0", "franka_follower_controllers.yaml"
    )

    leader_lqr = Node(
        package="franka_leader_3dof",
        executable="leader_fr3_j345_exact_lqr",
        namespace="leader",
        name="fr3_j345_exact_lqr",
        output="screen",
        parameters=[{
            "joint_name_prefix": "leader_fr3_",
            "joint_states_topic": "/leader/joint_states",
            "command_topic": "/leader/arm_3dof_cmd/commands",
        }],
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
