from setuptools import setup

package_name = 'da2_rgbd_publisher'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/da2_rtab.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Publish RGB + Depth from Depth Anything v2 to feed RTAB-Map',
    license='MIT',
    entry_points={
        'console_scripts': [
            'da2_rgbd_node = da2_rgbd_publisher.da2_rgbd_node:main',
        ],
    },
)
