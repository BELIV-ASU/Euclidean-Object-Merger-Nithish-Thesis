from setuptools import find_packages, setup

package_name = 'euclidean_object_merger'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nithish',
    maintainer_email='nithish@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['listener = euclidean_object_merger.subscriber:main',
        'transformer = euclidean_object_merger.frame_transform:main',
        'euclidean_object_merger_with_ego_vehicle_filter = euclidean_object_merger.euclidean_object_merger_with_ego_vehicle_filter:main',
        'object_merger_method2 = euclidean_object_merger.object_merger_method2:main',
        'object_merger_method2_version2 = euclidean_object_merger.object_merger_method2_version2:main'
        ],
    },
)
