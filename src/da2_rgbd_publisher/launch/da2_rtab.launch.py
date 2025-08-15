from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='da2_rgbd_publisher',
            executable='da2_rgbd_node',
            name='da2_rgbd',
            output='screen',
            parameters=[{
                'video_source': '/home/fliight/mp4_rgbd_ws/video.mp4',
                'resize_width': 320,
                'resize_height': 192,
                'hfov_deg': 70.0,
                'depth_min': 0.3,
                'depth_max': 8.0,
                'depth_scale': 1.0,
                'show_windows': False,
                'device': 'cpu'   # set 'cuda' if available
            }]
        ),
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='vo',
            output='screen',
            remappings=[
                ('rgb/image', '/camera/image_raw'),
                ('depth/image', '/camera/depth/image'),
                ('rgb/camera_info', '/camera/camera_info'),
                ('odom', '/odom'),
            ],
            parameters=[{
                'frame_id': 'camera_color_optical_frame',
                'odom_frame_id': 'odom',
                'approx_sync': True,
                'topic_queue_size': 10,
                'sync_queue_size': 50
            }]
        ),
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            remappings=[
                ('rgb/image', '/camera/image_raw'),
                ('depth/image', '/camera/depth/image'),
                ('rgb/camera_info', '/camera/camera_info'),
            ],
            parameters=[{
                'frame_id': 'camera_color_optical_frame',
                'odom_frame_id': 'odom',
                'subscribe_depth': True,
                'subscribe_rgb': True,
                'approx_sync': True,
                'sync_queue_size': 50,

                # build a 3D cloud/map from depth
                'Grid/FromDepth': True,         # build occupancy/grid from depth
                'Grid/3D': True,                # 3D grid enabled (also fine for 2D)
                'Grid/VoxelSize': 0.05,         # 5 cm voxels for /rtabmap/cloud_map
                # (optional) refresh rate
                'Rtabmap/DetectionRate': '10',
            }]
        ),

    ])
