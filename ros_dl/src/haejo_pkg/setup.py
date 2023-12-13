from setuptools import find_packages, setup
import os 
import glob

package_name = 'haejo_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*.launch.py')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='soomin',
    maintainer_email='isumin138@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'deep_learning = haejo_pkg.deep_learning:main',
            'usb_cam = usb_cam.usb_cam_node_exe:main'
        ],
    },
)
