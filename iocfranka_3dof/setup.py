from setuptools import setup
from glob import glob
import os

package_name = 'iocfranka_3dof'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='3-DOF Franka FR3 IOC control package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fr3_j345_exact_lqr = iocfranka_3dof.fr3_j345_exact_lqr:main',
            'fr3_j345_ramp_hold = iocfranka_3dof.fr3_j345_ramp_hold:main',
            'fr3_j345_hold_current = iocfranka_3dof.fr3_j345_hold_current:main',
            'fr3_j345_leader_style_lqr = iocfranka_3dof.fr3_j345_leader_style_lqr:main',
            'fr3_j345_torque_servo = iocfranka_3dof.fr3_j345_torque_servo:main',
            'fr3_j345_matlab_lqr_follower = iocfranka_3dof.fr3_j345_matlab_lqr_follower:main',
            'fr3_fm_leader_expert = iocfranka_3dof.fr3_fm_leader_expert:main',
            'fr3_fm_leader_collect = iocfranka_3dof.fr3_fm_leader_collect:main',
            'fr3_fm_offline_spsa = iocfranka_3dof.fr3_fm_offline_spsa:main',
            'fr3_fm_follower_run = iocfranka_3dof.fr3_fm_follower_run:main',
            'fr3_leader_torque_run = iocfranka_3dof.fr3_leader_torque_run:main',
            'fr3_fitK_torque = iocfranka_3dof.fr3_fitK_torque:main',
            'fr3_leader_torque_log = iocfranka_3dof.fr3_leader_torque_log:main',
            'fr3_j345_exact_lqr_follower = iocfranka_3dof.fr3_j345_exact_lqr_follower:main',
        ],
    },
)
