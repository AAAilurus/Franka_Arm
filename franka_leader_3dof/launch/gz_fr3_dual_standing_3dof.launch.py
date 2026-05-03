import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
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
        },
    )

    robot_description = {"robot_description": robot_description_config.toxml()}
    sim_time_param = {"use_sim_time": True}

    # Important:
    # Put robot_state_publisher inside the same namespace as the Gazebo plugin.
    # The plugin will look for /leader/leader_robot_state_publisher
    # and /follower/follower_robot_state_publisher.
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=namespace,
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

    return rsp, spawn


def spawners(namespace, pkg_share, hold_yaml, arm_yaml):
    cm = f"/{namespace}/controller_manager"

    jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "-c", cm,
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    hold = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "fixed_joints_hold",
            "-c", cm,
            "--param-file", os.path.join(pkg_share, "config", hold_yaml),
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_3dof_cmd",
            "-c", cm,
            "--param-file", os.path.join(pkg_share, "config", arm_yaml),
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return jsb, hold, arm



def controller_type_setters(namespace):
    cm = f"/{namespace}/controller_manager"
    return [
        ExecuteProcess(
            cmd=[
                "ros2", "param", "set", cm,
                "joint_state_broadcaster.type",
                "joint_state_broadcaster/JointStateBroadcaster",
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                "ros2", "param", "set", cm,
                "fixed_joints_hold.type",
                "joint_trajectory_controller/JointTrajectoryController",
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                "ros2", "param", "set", cm,
                "arm_3dof_cmd.type",
                "forward_command_controller/ForwardCommandController",
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                "ros2", "param", "set", cm,
                "use_sim_time",
                "true",
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                "ros2", "param", "set", cm,
                "update_rate",
                "1000",
            ],
            output="screen",
        ),
    ]


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_franka_desc = get_package_share_directory("franka_description")
    pkg_leader = get_package_share_directory("franka_leader_3dof")
    pkg_follower = get_package_share_directory("franka_follower_3dof")

    os.environ["GZ_SIM_RESOURCE_PATH"] = os.path.dirname(pkg_franka_desc)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": "empty.sdf -r"}.items(),
    )

    leader_rsp, leader_spawn = make_robot(
        "franka_leader_3dof",
        "leader",
        "leader_",
        "-0.35 0 0",
        "franka_leader_controllers.yaml",
    )

    follower_rsp, follower_spawn = make_robot(
        "franka_follower_3dof",
        "follower",
        "follower_",
        "0.35 0 0",
        "franka_follower_controllers.yaml",
    )

    leader_jsb, leader_hold, leader_arm = spawners(
        "leader",
        pkg_leader,
        "leader_hold.yaml",
        "leader_arm3.yaml",
    )

    follower_jsb, follower_hold, follower_arm = spawners(
        "follower",
        pkg_follower,
        "follower_hold.yaml",
        "follower_arm3.yaml",
    )

    leader_type_setters = controller_type_setters("leader")
    follower_type_setters = controller_type_setters("follower")

    return LaunchDescription([
        gazebo,

        leader_rsp,
        follower_rsp,

        TimerAction(period=3.0, actions=[leader_spawn]),
        TimerAction(period=7.0, actions=[follower_spawn]),

        TimerAction(period=12.0, actions=leader_type_setters),
        TimerAction(period=18.0, actions=[leader_jsb]),
        TimerAction(period=22.0, actions=[leader_hold]),
        TimerAction(period=26.0, actions=[leader_arm]),

        TimerAction(period=26.0, actions=follower_type_setters),
        TimerAction(period=32.0, actions=[follower_jsb]),
        TimerAction(period=36.0, actions=[follower_hold]),
        TimerAction(period=40.0, actions=[follower_arm]),
    ])
