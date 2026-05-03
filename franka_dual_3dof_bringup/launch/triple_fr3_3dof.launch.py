import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def make_robot(pkg_name, namespace, arm_prefix, xyz, controllers_file):
    pkg_share = get_package_share_directory(pkg_name)
    xacro_file = os.path.join(pkg_share, "urdf", "fr3_3dof.urdf.xacro")

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


def controller_type_setters(namespace):
    cm = f"/{namespace}/controller_manager"
    return [
        ExecuteProcess(cmd=["ros2", "param", "set", cm, "use_sim_time", "true"], output="screen"),
        ExecuteProcess(cmd=["ros2", "param", "set", cm, "update_rate", "1000"], output="screen"),
        ExecuteProcess(cmd=["ros2", "param", "set", cm, "joint_state_broadcaster.type", "joint_state_broadcaster/JointStateBroadcaster"], output="screen"),
        ExecuteProcess(cmd=["ros2", "param", "set", cm, "fixed_joints_hold.type", "joint_trajectory_controller/JointTrajectoryController"], output="screen"),
        ExecuteProcess(cmd=["ros2", "param", "set", cm, "arm_3dof_cmd.type", "forward_command_controller/ForwardCommandController"], output="screen"),
    ]


def spawners(namespace, hold_yaml, arm_yaml):
    share = get_package_share_directory("franka_dual_3dof_bringup")
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
            "--param-file", os.path.join(share, "config", hold_yaml),
        ],
        output="screen",
    )

    arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_3dof_cmd",
            "-c", cm,
            "--param-file", os.path.join(share, "config", arm_yaml),
        ],
        output="screen",
    )

    return jsb, hold, arm


def hold_command(namespace, prefix):
    topic = f"/{namespace}/fixed_joints_hold/joint_trajectory"
    return ExecuteProcess(
        cmd=[
            "ros2", "topic", "pub", "--once",
            topic,
            "trajectory_msgs/msg/JointTrajectory",
            "{"
            "joint_names: ['" + prefix + "fr3_joint1', '" + prefix + "fr3_joint2', '" + prefix + "fr3_joint6', '" + prefix + "fr3_joint7'], "
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
    pkg_leader = get_package_share_directory("franka_leader_3dof")
    pkg_follower = get_package_share_directory("franka_follower_3dof")
    pkg_bringup = get_package_share_directory("franka_dual_3dof_bringup")

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
        "-0.65 0.0 0.0",
        os.path.join(pkg_leader, "config", "franka_leader_controllers.yaml"),
    )

    follower_rsp, follower_spawn = make_robot(
        "franka_follower_3dof",
        "follower",
        "follower_",
        "0.65 0.0 0.0",
        os.path.join(pkg_follower, "config", "franka_follower_controllers.yaml"),
    )

    clone_rsp, clone_spawn = make_robot(
        "franka_leader_3dof",
        "clone",
        "clone_",
        "0.0 0.75 0.0",
        os.path.join(pkg_bringup, "config", "clone_controllers.yaml"),
    )

    leader2_rsp, leader2_spawn = make_robot(
        "franka_leader_3dof",
        "leader2",
        "leader2_",
        "0.0 -0.75 0.0",
        os.path.join(pkg_bringup, "config", "leader2_controllers.yaml"),
    )

    leader_jsb, leader_hold, leader_arm = spawners("leader", "leader_hold.yaml", "leader_arm3.yaml")
    follower_jsb, follower_hold, follower_arm = spawners("follower", "follower_hold.yaml", "follower_arm3.yaml")
    clone_jsb, clone_hold, clone_arm = spawners("clone", "clone_hold.yaml", "clone_arm3.yaml")
    leader2_jsb, leader2_hold, leader2_arm = spawners("leader2", "leader2_hold.yaml", "leader2_arm3.yaml")

    return LaunchDescription([
        gazebo,

        leader_rsp,
        follower_rsp,
        clone_rsp,
        leader2_rsp,

        TimerAction(period=3.0, actions=[leader_spawn]),
        TimerAction(period=6.0, actions=[follower_spawn]),
        TimerAction(period=9.0, actions=[clone_spawn]),
        TimerAction(period=12.0, actions=[leader2_spawn]),

        TimerAction(period=13.0, actions=controller_type_setters("leader")),
        TimerAction(period=15.0, actions=[leader_jsb]),
        TimerAction(period=17.0, actions=[leader_hold]),
        TimerAction(period=19.0, actions=[hold_command("leader", "leader_")]),
        TimerAction(period=21.0, actions=[leader_arm]),

        TimerAction(period=23.0, actions=controller_type_setters("follower")),
        TimerAction(period=25.0, actions=[follower_jsb]),
        TimerAction(period=27.0, actions=[follower_hold]),
        TimerAction(period=29.0, actions=[hold_command("follower", "follower_")]),
        TimerAction(period=31.0, actions=[follower_arm]),

        TimerAction(period=33.0, actions=controller_type_setters("clone")),
        TimerAction(period=35.0, actions=[clone_jsb]),
        TimerAction(period=37.0, actions=[clone_hold]),
        TimerAction(period=39.0, actions=[hold_command("clone", "clone_")]),
        TimerAction(period=41.0, actions=[clone_arm]),

        TimerAction(period=43.0, actions=controller_type_setters("leader2")),
        TimerAction(period=45.0, actions=[leader2_jsb]),
        TimerAction(period=47.0, actions=[leader2_hold]),
        TimerAction(period=49.0, actions=[hold_command("leader2", "leader2_")]),
        TimerAction(period=51.0, actions=[leader2_arm]),
    ])
