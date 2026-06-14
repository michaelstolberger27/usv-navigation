from setuptools import find_packages, setup

package_name = 'colav_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/demo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Michael Stolberger',
    maintainer_email='stolbergermichael5@gmail.com',
    description='ROS 2 node wrapping the COLAV hybrid-automaton controller.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'colav_node = colav_ros.colav_node:main',
            'fake_world = colav_ros.fake_world:main',
        ],
    },
)
