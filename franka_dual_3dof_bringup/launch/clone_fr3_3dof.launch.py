import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def make_robot(namespace, arm_prefix, xyz):
    # Important: use the working leader xacro structure,
    # but give it clone_ prefix and /clone namespace.
    pkg_share = get_package_share_directory("franka_leader_3dof")
    bringup_share = get_package_share_directory("franka_dual_3dof_bringup")

    xacro_file = os.path.join(pkg_share, "urdf", "fr3_3dof.urdf.xacro")
    controllers_file = os.path.join(bringup_share, "config", "clone_controllers.yaml")

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

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=namespace,
        name=f"{namespace}_robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}, robot_description],
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


def set_controller_types(namespace):
    cm = f"/{namespace}/controller_manager"
    return [
        ExecuteProcess(cmd=["ros2", "param", "set", cm, "use_sim_time", "true"], output="screen"),
        ExecuteProcess(cmd=["ros2", "param", "set", cm, "update_rate", "1000"], output="screen"),
        ExecuteProcess(cmd=["ros2", "param", "set", cm, "joint_state_broadcaster.type", "joint_state_broadcaster/JointStateBroadcaster"], output="screen"),
        ExecuteProcess(cmd=["ros2", "param", "set", cm, "fixed_joints_hold.type", "joint_trajectory_controller/JointTrajectoryController"], output="screen"),
        ExecuteProcess(cmd=["ros2", "param", "set", cm, "arm_3dof_cmd.type", "forward_command_controller/ForwardCommandController"], output="screen"),
    ]


def spawners(namespace):
    bringup_share = get_package_share_directory("franka_dual_3dof_bringup")
    cm = f"/{namespace}/controller_manager"

    jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", cm],
        output="screen",
    )

    hold = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "fixed_joints_hold",
            "-c", cm,
            "--param-file", os.path.join(bringup_share, "config", "clone_hold.yaml"),
        ],
        output="screen",
    )

    arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_3dof_cmd",
            "-c", cm,
            "--param-file", os.path.join(bringup_share, "config", "clone_arm3.yaml"),
        ],
        output="screen",
    )

    return jsb, hold, arm


def hold_command(namespace):
    topic = f"/{namespace}/fixed_joints_hold/joint_trajectory"
    return ExecuteProcess(
        cmd=[
            "ros2", "topic", "pub", "--once",
            topic,
            "trajectory_msgs/msg/JointTrajectory",
            "{"
            "joint_names: ['clone_fr3_joint1', 'clone_fr3_joint2', 'clone_fr3_joint6', 'clone_fr3_joint7'], "
            "points: [{"
            "positions: [0.0, -0.7853981634, 1.5707963268, 0.7853981634], "
            "velocities: [0.0, 0.0, 0.0, 0.0], "
            "time_from_start: {sec: 1, nanosec: 0}"
            "}]"
            "}",
        ],
        output="screen",
    )


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

    rsp, spawn = make_robot("clone", "clone_", "0.0 0.0 0.0")
    jsb, hold, arm = spawners("clone")

    return LaunchDescription([
        gazebo,
        rsp,
        TimerAction(period=3.0, actions=[spawn]),
        TimerAction(period=10.0, actions=set_controller_types("clone")),
        TimerAction(period=15.0, actions=[jsb]),
        TimerAction(period=18.0, actions=[hold]),
        TimerAction(period=20.0, actions=[hold_command("clone")]),
        TimerAction(period=22.0, actions=[arm]),
    ])
