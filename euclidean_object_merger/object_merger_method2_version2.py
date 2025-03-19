import rclpy
from rclpy.node import Node

from autoware_auto_perception_msgs.msg import DetectedObjects
from builtin_interfaces.msg import Time

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from shapely.geometry import Polygon
from shapely.affinity import rotate

import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import time

class ObjectMerger(Node):
    def __init__(self):
        super().__init__('object_merger_node')

        self.time_stamp = Time()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ego_vehicle_dx = 4.67
        self.ego_vehicle_dy = 1.81

        self.objects0_subscription = self.create_subscription(DetectedObjects,\
                                    '/perception/object_recognition/detection/centerpoint/objects', self.objects0_callback, 10)
        self.objects1_subscription = self.create_subscription(DetectedObjects,\
                                    '/perception/object_recognition/prediction/point/carom/objects', self.objects1_callback, 10)
        self.objects0 = DetectedObjects()
        self.objects1 = DetectedObjects()
        self.merged_objects_publisher = self.create_publisher(DetectedObjects, '/perception/object_recognition/detection/onboard_infrastructure_fused_objects', 10)

        self.timer = self.create_timer(0.1, self.object_merger_callback)

    def objects0_callback(self, objects0_msg):
        self.objects0 = objects0_msg

    def objects1_callback(self, objects1_msg):
        self.objects1 = objects1_msg
    
    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        
        return yaw_z # in radians

    def frame_transform_baselink_to_map(self):
        try:
            t1 = self.tf_buffer.lookup_transform('base_link', 'map', self.time_stamp)
            self.yaw_tf = self.euler_from_quaternion(t1.transform.rotation.x, t1.transform.rotation.y, t1.transform.rotation.z, t1.transform.rotation.w)
            self.x_tf, self.y_tf, self.z_tf = t1.transform.translation.x, t1.transform.translation.y, t1.transform.translation.z 
            self.ox, self.oy, self.oz, self.ow = t1.transform.rotation.x, t1.transform.rotation.y, t1.transform.rotation.z, t1.transform.rotation.w
        except TransformException as ex:
            self.get_logger().info(f'Could not transform "base_link" to "map": {ex}')
            return
        
    def frame_transform_map_to_baselink(self):
        try:
            t2 = self.tf_buffer.lookup_transform('map', 'base_link', self.time_stamp)
            self.ego_yaw_tf = self.euler_from_quaternion(t2.transform.rotation.x, t2.transform.rotation.y, t2.transform.rotation.z, t2.transform.rotation.w)
            self.ego_x_tf, self.ego_y_tf, self.ego_z_tf = t2.transform.translation.x, t2.transform.translation.y, t2.transform.translation.z 
            self.ego_ox, self.ego_oy, self.ego_oz, self.ego_ow = t2.transform.rotation.x, t2.transform.rotation.y, t2.transform.rotation.z, t2.transform.rotation.w
        except TransformException as ex:
            self.get_logger().info(f'Could not transform "map" to "base_link": {ex}')
            return
        
    def transform_object_map_to_baselink(self, vehicle_position_map_frame, vehicle_orientation_map_frame):
        self.frame_transform_baselink_to_map()
        vehicle_position_camera = np.array([vehicle_position_map_frame.x, vehicle_position_map_frame.y, vehicle_position_map_frame.z, 1])
        transformation_matrix = np.array([[np.cos(self.yaw_tf), -np.sin(self.yaw_tf), 0, self.x_tf],\
                                        [np.sin(self.yaw_tf), np.cos(self.yaw_tf), 0, self.y_tf],\
                                        [0, 0, 1, self.z_tf],\
                                        [0, 0, 0, 1]])
        transformed_position = np.dot(transformation_matrix, vehicle_position_camera.T).T
        vehicle_orientation_camera = np.array([vehicle_orientation_map_frame.x, vehicle_orientation_map_frame.y,\
                                                vehicle_orientation_map_frame.z, vehicle_orientation_map_frame.w])
        vehicle_orientation_base = np.array([self.ox, self.oy, self.oz, self.ow])
        r1 = R.from_quat(vehicle_orientation_camera)
        r2 = R.from_quat(vehicle_orientation_base)
        r3 = r2*r1
        transformed_orientation = r3.as_quat()

        return transformed_position, transformed_orientation
    
    def create_rotated_box(self, center_x, center_y, width, height, angle):
        half_width = width / 2
        half_height = height / 2
        rectangle = Polygon([
            (center_x - half_width, center_y - half_height),
            (center_x + half_width, center_y - half_height),
            (center_x + half_width, center_y + half_height),
            (center_x - half_width, center_y + half_height)
        ])
        rotated_rectangle = rotate(rectangle, angle, origin='center', use_radians=False)
        return rotated_rectangle
    
    def calculate_iou(self, box0, box1):
        intersection = box0.intersection(box1).area
        union = box0.union(box1).area
        return intersection / union if union > 0 else 0

    def object_merger_callback(self):
        self.start_time = time.time()
        #self.get_logger().info('Entering object merger callback')

        # Update the self.time_stamp get the transformation from base_link to map frame at that specific instance
        self.time_stamp = self.objects0.header.stamp

        # create a cache of object0 so that object1 info can be added to this cache if needed
        self.objects_cache = self.objects0
        self.objects_cache.header.stamp = self.time_stamp
        

        for obj1 in self.objects1.objects:
            # initialize the overlap flag
            overlap_flag = 0

            # For removing ego vehicle from being merged
            self.frame_transform_map_to_baselink()
            box0 = self.create_rotated_box(self.ego_x_tf, self.ego_y_tf,\
                    self.ego_vehicle_dx, self.ego_vehicle_dy,\
                    np.rad2deg(self.euler_from_quaternion(self.ego_ox, self.ego_oy, self.ego_oz, self.ego_ow)))
            box1 = self.create_rotated_box(obj1.kinematics.pose_with_covariance.pose.position.x, obj1.kinematics.pose_with_covariance.pose.position.y,\
                            obj1.shape.dimensions.x, obj1.shape.dimensions.y,\
                            np.rad2deg(self.euler_from_quaternion(obj1.kinematics.pose_with_covariance.pose.orientation.x, obj1.kinematics.pose_with_covariance.pose.orientation.y,\
                            obj1.kinematics.pose_with_covariance.pose.orientation.z, obj1.kinematics.pose_with_covariance.pose.orientation.w)))
            iou = self.calculate_iou(box0, box1)
        
            if iou>0.0:
                print("Ego vehicle found")
                continue
                    

            # transform object from map frame to base_link
            [obj1.kinematics.pose_with_covariance.pose.position.x,\
                obj1.kinematics.pose_with_covariance.pose.position.y,\
                obj1.kinematics.pose_with_covariance.pose.position.z, scale], \
                [obj1.kinematics.pose_with_covariance.pose.orientation.x, \
                obj1.kinematics.pose_with_covariance.pose.orientation.y, \
                obj1.kinematics.pose_with_covariance.pose.orientation.z, \
                obj1.kinematics.pose_with_covariance.pose.orientation.w]  = \
                    self.transform_object_map_to_baselink(obj1.kinematics.pose_with_covariance.pose.position,\
                                                    obj1.kinematics.pose_with_covariance.pose.orientation)
            box1 = self.create_rotated_box(obj1.kinematics.pose_with_covariance.pose.position.x, obj1.kinematics.pose_with_covariance.pose.position.y,\
                            obj1.shape.dimensions.x, obj1.shape.dimensions.y,\
                            np.rad2deg(self.euler_from_quaternion(obj1.kinematics.pose_with_covariance.pose.orientation.x, obj1.kinematics.pose_with_covariance.pose.orientation.y,\
                            obj1.kinematics.pose_with_covariance.pose.orientation.z, obj1.kinematics.pose_with_covariance.pose.orientation.w)))
            
            for obj0 in self.objects0.objects:
                box0 = self.create_rotated_box(obj0.kinematics.pose_with_covariance.pose.position.x, obj0.kinematics.pose_with_covariance.pose.position.y,\
                            obj0.shape.dimensions.x, obj0.shape.dimensions.y,\
                            np.rad2deg(self.euler_from_quaternion(obj0.kinematics.pose_with_covariance.pose.orientation.x, obj0.kinematics.pose_with_covariance.pose.orientation.y,\
                            obj0.kinematics.pose_with_covariance.pose.orientation.z, obj0.kinematics.pose_with_covariance.pose.orientation.w)))
                
                iou = self.calculate_iou(box0, box1)
                
                if iou>0.05:
                    overlap_flag = 1
                    break
            
            if overlap_flag == 0:
                self.objects_cache.objects.append(obj1)
        
        self.objects0 = DetectedObjects()
        self.objects1 = DetectedObjects()
        self.merged_objects_publisher.publish(self.objects_cache)
        self.objects_cache = DetectedObjects()
        time_difference = time.time() - self.start_time
        #print("Time taken for one iteration of object merger: ", time_difference*1000, "ms")
            
        

def main(args=None):
    rclpy.init(args=args)
    node = ObjectMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
