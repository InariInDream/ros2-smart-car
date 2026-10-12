import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory("line_follower")
    default_params = os.path.join(pkg_share, "config", "line_follower_params.yaml")

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/color/image_raw",
        description="巡线订阅的RGB图像话题",
    )
    cmd_vel_topic_arg = DeclareLaunchArgument(
        "cmd_vel_topic",
        default_value="/cmd_vel",
        description="底盘最终速度话题（mux 输出）",
    )
    follower_cmd_arg = DeclareLaunchArgument(
        "follower_cmd_topic",
        default_value="/cmd_vel_follower",
        description="巡线节点输出话题",
    )
    detection_cmd_arg = DeclareLaunchArgument(
        "detection_cmd_topic",
        default_value="/cmd_vel_detection",
        description="AI检测节点输出话题",
    )
    stop_cmd_arg = DeclareLaunchArgument(
        "stop_cmd_topic",
        default_value="/cmd_vel_stop",
        description="巡线停靠输出的停止话题（最高优先级）",
    )
    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="参数文件",
    )

    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    follower_cmd_topic = LaunchConfiguration("follower_cmd_topic")
    detection_cmd_topic = LaunchConfiguration("detection_cmd_topic")
    stop_cmd_topic = LaunchConfiguration("stop_cmd_topic")
    params_file = LaunchConfiguration("params_file")

    # 巡线节点 - 输出到follower_cmd_topic
    line_follower_node = Node(
        package="line_follower",
        executable="line_follower_node",
        name="line_follower_node",
        output="screen",
        parameters=[
            params_file,
            {
                "cmd_vel_topic": follower_cmd_topic,
            },
        ],
    )

    # YOLO检测转向控制器（使用原有的detection_turn_controller，已适配vision_msgs）
    detection_turn_node = Node(
        package="line_follower",
        executable="detection_turn_controller",
        name="detection_turn_controller",
        output="screen",
        parameters=[
            params_file,
            {
                "cmd_vel_topic": detection_cmd_topic,
            },
        ],
    )

    # 命令多路复用节点 - 停止 > AI检测 > 巡线
    mux_node = Node(
        package="line_follower",
        executable="cmd_mux_node",
        name="cmd_mux_node",
        output="screen",
        parameters=[
            params_file,
            {
                "stop_topic": stop_cmd_topic,
                "stop_timeout": 0.5,
                "high_topic": detection_cmd_topic,  # YOLO检测（最高优先级）
                "high_timeout": 2.0,
                "low_topic": follower_cmd_topic,   # 巡线（低优先级）
                "output_topic": cmd_vel_topic,
            },
        ],
    )

    return LaunchDescription([
        image_topic_arg,
        cmd_vel_topic_arg,
        follower_cmd_arg,
        detection_cmd_arg,
        stop_cmd_arg,
        params_arg,
        line_follower_node,
        detection_turn_node,
        mux_node,
    ])
