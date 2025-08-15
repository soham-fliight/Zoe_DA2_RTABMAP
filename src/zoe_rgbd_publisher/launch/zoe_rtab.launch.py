from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    video = LaunchConfiguration('video')  # <-- new
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'video', default_value='',
            description='Absolute path to MP4 (or camera index)'
        ),


        # ZoeDepth RGB-D from MP4
        Node(
            package='zoe_rgbd_publisher',
            executable='zoe_rgbd_node',
            name='zoe_rgbd',
            output='screen',
            parameters=[{
                'video_source': video,                # <-- use LaunchConfiguration, not ""
                'img_size_w': 640, 'img_size_h': 480,
                'frame_id': 'camera_color_optical_frame',
                'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0,
                'fps': 30.0,
                'depth_scale': 1.0
            }]
        ),

        # RGB-D odometry + mapping (RTAB-Map)
        Node(
            package='rtabmap_odom',
            executable='rgbd_odometry',
            name='vo',
            output='screen',
            parameters=[{
                'frame_id':'camera_color_optical_frame',
                'approx_sync': True,
                'sync_queue_size': 20,        # (renamed from queue_size)
                'subscribe_rgbd': False,      # <-- use separate topics
                'subscribe_depth': True       # <-- enable RGB-D mode
                # 'Vis/MinInliers': '15'      # optional, as string; or omit
            }],
            remappings=[('rgb/image','/camera/image_raw'),
                        ('depth/image','/camera/depth/image'),
                        ('rgb/camera_info','/camera/camera_info'),
                        ('odom','/odom')]
        ),

        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[{
                'subscribe_rgbd': False,      # <-- use separate topics
                'subscribe_depth': True,      # <-- RGB  Depth  CameraInfo
                'approx_sync': True,
                'sync_queue_size': 20,
                'frame_id':'camera_color_optical_frame',
                'map_frame_id':'map',
                'odom_frame_id':'odom',
                'Rtabmap/DetectionRate':'10',
                'Mem/IncrementalMemory':'true'
            }],
            remappings=[('rgb/image','/camera/image_raw'),
                        ('depth/image','/camera/depth/image'),
                        ('rgb/camera_info','/camera/camera_info'),
                        ('odom','/odom')]
        ),

    ])
