import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_ioc = get_package_share_directory("franka_follower_3dof")
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_franka_desc = get_package_share_directory("franka_description")

    os.environ["GZ_SIM_RESOURCE_PATH"] = os.path.dirname(pkg_franka_desc)

    xacro_file = os.path.join(pkg_ioc, "urdf", "fr3_3dof.urdf.xacro")
    robot_description_config = xacro.process_file(xacro_file)
    robot_description = {"robot_description": robot_description_config.toxml()}

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": "empty.sdf -r"}.items(),
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-topic", "/robot_description"],
        output="screen",
    )

    rviz_file = os.path.join(pkg_franka_desc, "rviz", "visualize_franka.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["--display-config", rviz_file, "-f", "world"],
        output="screen",
    )

    load_jsb = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-state", "active", "joint_state_broadcaster"],
        output="screen",
    )

    load_hold = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-state", "active", "fixed_joints_hold"],
        output="screen",
    )

    load_arm = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-state", "active", "arm_3dof_cmd"],
        output="screen",
    )

    send_hold_pose = ExecuteProcess(
        cmd=[
            "ros2", "topic", "pub", "--once",
            "/fixed_joints_hold/joint_trajectory",
            "trajectory_msgs/msg/JointTrajectory",
            '{"joint_names":["fr3_joint1","fr3_joint2","fr3_joint6","fr3_joint7"],"points":[{"positions":[0.0,-0.7853981634,1.5707963268,0.7853981634],"time_from_start":{"sec":2}}]}'
        ],
        output="screen",
    )

    return LaunchDescription([
        gazebo,
        rsp,
        rviz,
        spawn,
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn,
                on_exit=[load_jsb],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_jsb,
                on_exit=[load_hold],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_hold,
                on_exit=[load_arm, TimerAction(period=2.0, actions=[send_hold_pose])],
            )
        ),
    ])
