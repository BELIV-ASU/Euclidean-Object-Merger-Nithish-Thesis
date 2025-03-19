import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import pandas as pd
import numpy as np
import math

from autoware_auto_perception_msgs.msg import DetectedObjects
from autoware_auto_perception_msgs.msg import DetectedObject
from autoware_auto_perception_msgs.msg import ObjectClassification

class AutowareObjectsSubscriberPublisher(Node):
	def __init__(self):
		super().__init__('autoware_objects_subscriber_publisher')
		self.entry_columns = ['existence_probability', 'source', 'object_class', 'x', 'y', 'z', 'o_x', 'o_y', 'o_z', 'o_w', 'd_x', 'd_y', 'd_z']
		
		self.onboard_objects_df = pd.DataFrame(columns=self.entry_columns)
		self.infrastructure_objects_df = pd.DataFrame(columns=self.entry_columns)
		self.merged_objects_df = pd.DataFrame(columns=self.entry_columns)
		self.onboard_objects = self.create_subscription(DetectedObjects, '/perception/object_recognition/detection/centerpoint/objects', self.onboard_objects_df_converter, 10)
		self.infrastructure_objects = self.create_subscription(DetectedObjects, '/perception/object_recognition/prediction/point/carom/objects', self.infrastructure_objects_df_converter, 10)
		self.fused_objects_publisher = self.create_publisher(DetectedObjects, '/perception/object_recognition/detection/onboard_infrastructure_fused_objects', 10)

		self.columns = ['cx1', 'cy1', 'cx2', 'cy2', 'cx3', 'cy3', 'cx4', 'cy4', 'cx5', 'cy5', 'existence_probability', 'source', 'object_class', 'x', 'y', 'z', 'o_x', 'o_y', 'o_z', 'o_w',
       'd_x', 'd_y', 'd_z', 'duplicate']
		self.objects_df = pd.DataFrame(columns=self.columns)

		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.timer1 = self.create_timer(0.01, self.df_merger)
		self.timer2 = self.create_timer(0.005, self.frame_transform)

		self.x_tf = 0
		self.y_tf = 0
		self.z_tf = 0
		self.roll_tf = 0
		self.pitch_tf = 0
		self.yaw_tf = 0

		self.header_info = 0

	def objects_publisher(self):
		publish_objects_msg = DetectedObjects()
		publish_objects_msg.header =  self.header_info
		fusion_objects_df_index = [p for p in range(len(self.objects_df)) if self.objects_df.iloc[p]['duplicate']==0.0]
		number_of_objects_after_fusion = len(fusion_objects_df_index)
		objects = [0 for o in range(number_of_objects_after_fusion)]
		classification = [0]
		for n in range(number_of_objects_after_fusion):
			objects[n] = DetectedObject()
			classification[0] = ObjectClassification()
			objects[n].existence_probability = self.objects_df.iloc[fusion_objects_df_index[n]]['existence_probability']
			classification[0].label = int(self.objects_df.iloc[fusion_objects_df_index[n]]['object_class'])
			#classification[0].label = 1 # Make it simpler for now. Label 1 corresponds to car.
			objects[n].classification = list(classification)
			objects[n].kinematics.pose_with_covariance.pose.position.x = self.objects_df.iloc[fusion_objects_df_index[n]]['x']
			objects[n].kinematics.pose_with_covariance.pose.position.y = self.objects_df.iloc[fusion_objects_df_index[n]]['y']
			objects[n].kinematics.pose_with_covariance.pose.position.z = self.objects_df.iloc[fusion_objects_df_index[n]]['z']
			objects[n].kinematics.pose_with_covariance.pose.orientation.x = self.objects_df.iloc[fusion_objects_df_index[n]]['o_x']
			objects[n].kinematics.pose_with_covariance.pose.orientation.y = self.objects_df.iloc[fusion_objects_df_index[n]]['o_y']
			objects[n].kinematics.pose_with_covariance.pose.orientation.z = self.objects_df.iloc[fusion_objects_df_index[n]]['o_z']
			objects[n].kinematics.pose_with_covariance.pose.orientation.w = self.objects_df.iloc[fusion_objects_df_index[n]]['o_w']
			objects[n].kinematics.orientation_availability = 1
			objects[n].shape.dimensions.x = self.objects_df.iloc[fusion_objects_df_index[n]]['d_x']
			objects[n].shape.dimensions.y = self.objects_df.iloc[fusion_objects_df_index[n]]['d_y']
			objects[n].shape.dimensions.z = self.objects_df.iloc[fusion_objects_df_index[n]]['d_z']

		publish_objects_msg.objects = objects
		self.fused_objects_publisher.publish(publish_objects_msg)

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

	def euler_to_quaternion(self, roll, pitch, yaw):
		cy = np.cos(yaw * 0.5)
		sy = np.sin(yaw * 0.5)
		cp = np.cos(pitch * 0.5)
		sp = np.sin(pitch * 0.5)
		cr = np.cos(roll * 0.5)
		sr = np.sin(roll * 0.5)

		w = cr * cp * cy + sr * sp * sy
		x = sr * cp * cy - cr * sp * sy
		y = cr * sp * cy + sr * cp * sy
		z = cr * cp * sy - sr * sp * cy

		return [x, y, z, w]

	def transform_lidar_frame_x_to_base_frame_x(self, x, y, z, tx, yaw_angle):
		x_ = x*math.cos(yaw_angle) - y*math.sin(yaw_angle) + tx
		#print('x:',x, 'yaw_angle:', yaw_angle, 'x_', x_)
		return x_

	def transform_lidar_frame_y_to_base_frame_y(self, x, y, z, ty, yaw_angle):
		y_ = x*math.sin(yaw_angle) + y*math.cos(yaw_angle) + ty
		return y_

	def transform_lidar_frame_z_to_base_frame_z(self, x, y, z, tz, yaw_angle):
		z_ = z + tz
		return z_

	def find_centerline_points(self, p_x, p_y, c_y ,yaw_angle):
		centerline_x = (c_y) * math.cos(yaw_angle*math.pi/180)
		centerline_y = (c_y) * math.sin(yaw_angle*math.pi/180)
		centerline_x = p_x + centerline_x
		centerline_y = p_y + centerline_y
		return centerline_x, centerline_y

	def find_centerline_points_for_overlap_verification(self, p_x, p_y, d_x, d_y, yaw_angle):
		points = []
		box_ratio = round((d_x/d_y),1)
		centerline_points = []

		centerline_points.append([p_x, p_y])

		if box_ratio <= 2.5:
			points = [d_x/2 - (d_y*0.5), -d_x/2 - (-d_y*0.5)]
		elif box_ratio > 2.5:
			points = [d_x/2 - (d_y*0.5), -d_x/2 - (-d_y*0.5), d_x/2 - (d_y), -d_x/2 - (-d_y)]

		for i in range(4):
			if i>=len(points):
				centerline_points.append([np.nan, np.nan])
			else:
				centerline_points.append(self.find_centerline_points(p_x, p_y, points[i], yaw_angle))

		return centerline_points

	def frame_transform(self):
		try:
			#t = self.tf_buffer.lookup_transform('base_link', 'lidar_infrastructure', rclpy.time.Time())
			t = self.tf_buffer.lookup_transform('base_link', 'map', rclpy.time.Time())
			self.yaw_tf = self.euler_from_quaternion(t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
			self.x_tf, self.y_tf, self.z_tf = t.transform.translation.x, t.transform.translation.y, t.transform.translation.z 
			#print('yaw_tf', self.yaw_tf)
		except TransformException as ex:
			self.get_logger().info(f'Could not transform "base_link" to "map": {ex}')
			return
		
	def df_merger(self):
		if not(self.onboard_objects_df.empty) and not(self.infrastructure_objects_df.empty):
			#print(self.merged_objects_df)
			#x_, y_, z_ = self.infrastructure_objects_df['x'], self.infrastructure_objects_df['y'], self.infrastructure_objects_df['z']
			self.infrastructure_objects_df['x_new'] = self.infrastructure_objects_df.apply(lambda row: self.transform_lidar_frame_x_to_base_frame_x(row['x'],row['y'],row['z'], self.x_tf, self.yaw_tf), axis=1)
			self.infrastructure_objects_df['y_new'] = self.infrastructure_objects_df.apply(lambda row: self.transform_lidar_frame_y_to_base_frame_y(row['x'],row['y'],row['z'], self.y_tf, self.yaw_tf), axis=1)
			self.infrastructure_objects_df['z_new'] = self.infrastructure_objects_df.apply(lambda row: self.transform_lidar_frame_z_to_base_frame_z(row['x'],row['y'],row['z'], self.z_tf, self.yaw_tf), axis=1)
			self.infrastructure_objects_df['x'] = self.infrastructure_objects_df['x_new']
			self.infrastructure_objects_df['y'] = self.infrastructure_objects_df['y_new']
			self.infrastructure_objects_df['z'] = self.infrastructure_objects_df['z_new']
			self.infrastructure_objects_df['vehicle_angle'] = self.infrastructure_objects_df.apply(lambda row: self.euler_from_quaternion(row['o_x'], row['o_y'], row['o_z'], row['o_w']), axis=1)
			transformed_quaternion = self.infrastructure_objects_df.apply(lambda row: self.euler_to_quaternion(0, 0, row['vehicle_angle']+self.yaw_tf), axis=1)
			transformed_quaternion_x_array, transformed_quaternion_y_array,transformed_quaternion_z_array,transformed_quaternion_w_array = [], [], [], []
			for m in range(len(transformed_quaternion)):
				transformed_quaternion_x_array.append(transformed_quaternion[m][0])
				transformed_quaternion_y_array.append(transformed_quaternion[m][1])
				transformed_quaternion_z_array.append(transformed_quaternion[m][2])
				transformed_quaternion_w_array.append(transformed_quaternion[m][3])

			#print("quaternion_array: ", transformed_quaternion_x_array, transformed_quaternion_y_array,transformed_quaternion_z_array,transformed_quaternion_w_array)
			transformed_quaternion_x = pd.Series(transformed_quaternion_x_array)
			transformed_quaternion_y = pd.Series(transformed_quaternion_y_array)
			transformed_quaternion_z = pd.Series(transformed_quaternion_z_array)
			transformed_quaternion_w = pd.Series(transformed_quaternion_w_array)

			self.infrastructure_objects_df['o_x'], self.infrastructure_objects_df['o_y'], self.infrastructure_objects_df['o_z'], self.infrastructure_objects_df['o_w'] = transformed_quaternion_x, transformed_quaternion_y, transformed_quaternion_z, transformed_quaternion_w

			#print("transformed quaternion: ", transformed_quaternion_x, transformed_quaternion_y, transformed_quaternion_z, transformed_quaternion_w)
			self.infrastructure_objects_df = self.infrastructure_objects_df.drop(['x_new', 'y_new', 'z_new', 'vehicle_angle'], axis='columns')
			self.merged_objects_df = self.infrastructure_objects_df.append(self.onboard_objects_df, ignore_index=True)
			#print("merged_objects_df", self.merged_objects_df)
			#print("infrastructure_objects_df", self.infrastructure_objects_df)
			#print("onboard_objects_df", self.onboard_objects_df)

			for val in range(len(self.merged_objects_df)):
				source = self.merged_objects_df['source'][val]
				object_class = self.merged_objects_df['object_class'][val]
				p_x = self.merged_objects_df['x'][val]
				p_y = self.merged_objects_df['y'][val]
				p_z = self.merged_objects_df['z'][val]
				o_x = self.merged_objects_df['o_x'][val]
				o_y = self.merged_objects_df['o_y'][val]
				o_z = self.merged_objects_df['o_z'][val]
				o_w = self.merged_objects_df['o_w'][val]
				d_x = self.merged_objects_df['d_x'][val]
				d_y = self.merged_objects_df['d_y'][val]
				d_z = self.merged_objects_df['d_z'][val]

				vehicle_angle = self.euler_from_quaternion(o_x, o_y, o_z, o_w)
				#print("vehicle_angle: ", vehicle_angle)
				yaw_angle = vehicle_angle * 180 / math.pi

				centerpoints = self.find_centerline_points_for_overlap_verification(p_x, p_y, d_x, d_y, yaw_angle)

				centerpoints.append(list(self.merged_objects_df.iloc[val]))
				centerpoints = np.concatenate(centerpoints)
				centerpoints = np.append(centerpoints, [0])
				centerpoints_dict = dict(zip(self.columns, list(centerpoints)))
				self.objects_df = self.objects_df.append(centerpoints_dict, ignore_index=True)

			number_of_onboard_objects = np.count_nonzero(self.objects_df['source']==0)
			for i in range(len(self.objects_df)):
				if self.objects_df.iloc[i]['source']==0:
					#print(i)
					x1 = self.objects_df.iloc[i]['cx1']
					y1 = self.objects_df.iloc[i]['cy1']
					r1 = self.objects_df.iloc[i]['d_y']/2
					for j in range(len(self.objects_df)- number_of_onboard_objects):
						#print(j)
						if self.objects_df.iloc[j]['duplicate'] != 1:
							#and (self.objects_df.iloc[i]['object_class']==self.objects_df.iloc[j]['object_class']):
							x2 = self.objects_df.iloc[j]['cx1']
							y2 = self.objects_df.iloc[j]['cy1']
							r2 = self.objects_df.iloc[j]['d_y']/2
							euclidean_distance = math.sqrt(pow((x2-x1), 2) + pow((y2-y1), 2))
							if(euclidean_distance < (self.objects_df.iloc[i]['d_x']/2 + self.objects_df.iloc[j]['d_x']/2)):
								onboard_objects_df_row = self.objects_df.loc[i]
								number_of_centerline_points_in_onboard_object = len(onboard_objects_df_row) - np.count_nonzero(np.isnan(onboard_objects_df_row)) - 13
								infrastructure_objects_df_row = self.objects_df.loc[j]
								number_of_centerline_points_in_onboard_object = len(infrastructure_objects_df_row) - np.count_nonzero(np.isnan(infrastructure_objects_df_row)) -13
								for k in range(int(number_of_centerline_points_in_onboard_object/2)):
									if self.objects_df.iloc[j]['duplicate'] == 1.0:
										#print("In four")
										break
									x1 = self.objects_df.iloc[i]['cx'+str(k+1)]
									y1 = self.objects_df.iloc[i]['cy'+str(k+1)]
									#print("In three")
									for l in range(int(number_of_centerline_points_in_onboard_object/2)):
										x2 = self.objects_df.iloc[j]['cx'+str(l+1)]
										y2 = self.objects_df.iloc[j]['cy'+str(l+1)]
										euclidean_distance = math.sqrt(pow((x2-x1), 2) + pow((y2-y1), 2))
										#print("In one")
										if euclidean_distance < (r1 + r2):
											#print("In two")
											self.objects_df.loc[j, 'duplicate'] = 1.0
											break
			#print(self.objects_df['object_class'])
			self.objects_publisher()
			self.objects_df = pd.DataFrame(columns=self.columns)
			self.onboard_objects_df = pd.DataFrame(columns=self.entry_columns)
			self.infrastructure_objects_df = pd.DataFrame(columns=self.entry_columns)
			#exit()

	def onboard_objects_df_converter(self, onboard_detection_msg):
		#self.get_logger().info('I heard: "%s"' %len(onboard_detection_msg.objects[0].kinematics.pose_with_covariance.covariance))
		#self.get_logger().info('Number of objects: "%s"' %len(onboard_detection_msg.objects))
		self.header_info = onboard_detection_msg.header
		for i in range(len(onboard_detection_msg.objects)):
			onboard_object_info_autoware_format = onboard_detection_msg.objects[i]
			onboard_object_info = [onboard_object_info_autoware_format.existence_probability, 0,
							onboard_object_info_autoware_format.classification[0].label, 
							onboard_object_info_autoware_format.kinematics.pose_with_covariance.pose.position.x,
							onboard_object_info_autoware_format.kinematics.pose_with_covariance.pose.position.y,
							onboard_object_info_autoware_format.kinematics.pose_with_covariance.pose.position.z,
							onboard_object_info_autoware_format.kinematics.pose_with_covariance.pose.orientation.x,
							onboard_object_info_autoware_format.kinematics.pose_with_covariance.pose.orientation.y,
							onboard_object_info_autoware_format.kinematics.pose_with_covariance.pose.orientation.z,
							onboard_object_info_autoware_format.kinematics.pose_with_covariance.pose.orientation.w,
							onboard_object_info_autoware_format.shape.dimensions.x,
							onboard_object_info_autoware_format.shape.dimensions.y,
							onboard_object_info_autoware_format.shape.dimensions.z]
			onboard_object_info_dict = dict(zip(self.entry_columns, list(onboard_object_info)))
			self.onboard_objects_df = self.onboard_objects_df.append(onboard_object_info_dict, ignore_index=True)
			#print(self.onboard_objects_df.head())
		#exit()

	def infrastructure_objects_df_converter(self, infrastructure_detection_msg):
		#self.get_logger().info('I heard: "%s"' %infrastructure_detection_msg.objects[0].classification)
		#self.get_logger().info('Number of objects: "%s"' %len(infrastructure_detection_msg.objects))
		for j in range(len(infrastructure_detection_msg.objects)):
			infrastructure_object_info_autoware_format = infrastructure_detection_msg.objects[j]
			infrastructure_object_info = [infrastructure_object_info_autoware_format.existence_probability, 1,
							infrastructure_object_info_autoware_format.classification[0].label, 
							infrastructure_object_info_autoware_format.kinematics.pose_with_covariance.pose.position.x,
							infrastructure_object_info_autoware_format.kinematics.pose_with_covariance.pose.position.y,
							infrastructure_object_info_autoware_format.kinematics.pose_with_covariance.pose.position.z,
							infrastructure_object_info_autoware_format.kinematics.pose_with_covariance.pose.orientation.x,
							infrastructure_object_info_autoware_format.kinematics.pose_with_covariance.pose.orientation.y,
							infrastructure_object_info_autoware_format.kinematics.pose_with_covariance.pose.orientation.z,
							infrastructure_object_info_autoware_format.kinematics.pose_with_covariance.pose.orientation.w,
							infrastructure_object_info_autoware_format.shape.dimensions.x,
							infrastructure_object_info_autoware_format.shape.dimensions.y,
							infrastructure_object_info_autoware_format.shape.dimensions.z]
			infrastructure_object_info_dict = dict(zip(self.entry_columns, list(infrastructure_object_info)))
			self.infrastructure_objects_df = self.infrastructure_objects_df.append(infrastructure_object_info_dict, ignore_index=True)
			#print(self.infrastructure_objects_df.head())
		self.df_merger()
		#exit()


def main(args=None):
	rclpy.init(args=args)
	subscriber_publisher = AutowareObjectsSubscriberPublisher()
	rclpy.spin(subscriber_publisher)
	subscriber_publisher.destroy_node()
	rclpy.shutdown()
	
if __name__ == '__main__':
	main()
    
