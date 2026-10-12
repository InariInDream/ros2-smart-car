from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    selected_map = LaunchConfiguration('map').perform(context)
    vision_map = {
        '1': 'vision_node',
        '2': 'vision_node_2',
        '3': 'vision_node_3',
        'vision_node': 'vision_node',
        'vision_node_2': 'vision_node_2',
        'vision_node_3': 'vision_node_3',
    }

    if selected_map not in vision_map:
        raise RuntimeError(
            'map 只支持 1/2/3 或 vision_node/vision_node_2/vision_node_3'
        )

    vision_executable = vision_map[selected_map]
    vision_delay = float(LaunchConfiguration('vision_delay').perform(context))
    yolo_delay = float(LaunchConfiguration('yolo_delay').perform(context))

    camera_node = Node(
        package='yolo_cam',
        executable='camera_node',
        name='camera_node',
        output='screen'
    )

    vision_node = TimerAction(
        period=vision_delay,
        actions=[
            Node(
                package='vision_node',
                executable=vision_executable,
                name='vision_node',
                output='screen'
            )
        ]
    )

    yolo_node = TimerAction(
        period=yolo_delay,
        actions=[
            Node(
                package='yolo_cam',
                executable='yolo_node',
                name='yolo_node',
                output='screen',
                parameters=[LaunchConfiguration('yolo_config')]
            )
        ]
    )

    return [camera_node, vision_node, yolo_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value='1',
            description='选择地图对应的 vision 节点: 1, 2, 3 或直接传 executable 名称'
        ),
        DeclareLaunchArgument(
            'yolo_config',
            default_value=PathJoinSubstitution(
                [FindPackageShare('yolo_cam'), 'config', 'yolo_node.yaml']
            ),
            description='yolo_node 参数文件路径'
        ),
        DeclareLaunchArgument(
            'vision_delay',
            default_value='1.0',
            description='camera_node 启动后延迟多少秒再启动 vision 节点'
        ),
        DeclareLaunchArgument(
            'yolo_delay',
            default_value='2.0',
            description='camera_node 启动后延迟多少秒再启动 yolo 节点'
        ),
        OpaqueFunction(function=launch_setup),
    ])
