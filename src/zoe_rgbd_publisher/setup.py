from setuptools import setup
setup(
    name='zoe_rgbd_publisher',
    version='0.1.0',
    packages=['zoe_rgbd_publisher'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/zoe_rgbd_publisher']),
        ('share/zoe_rgbd_publisher', ['package.xml']),
        ('share/zoe_rgbd_publisher/launch', ['launch/zoe_rtab.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Publish RGB + ZoeDepth depth + CameraInfo from an MP4.',
    entry_points={'console_scripts': ['zoe_rgbd_node = zoe_rgbd_publisher.zoe_rgbd_node:main']},
)
