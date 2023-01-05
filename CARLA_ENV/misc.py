import math
import numpy as np
import carla
import pygame
from matplotlib.path import Path
import skimage

def get_speed(vehicle):
	"""
	argument: carla vehicle object
	return: speed of the vehicle in Kph
	"""
	vel = vehicle.get_velocity()
	return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
	
def get_pos(vehicle):
	"""
	argument: carla vehicle object
	return: x y coordinates
	"""
	trans = vehicle.get_transform()
	return trans.location.x, trans.location.y
	
def get_info(vehicle):
	"""
	argument: carla vehicle object
	return: x y coordinates, heading(yaw) angle and half length width of the vehicle bounding box
	"""
	trans = vehicle.get_transform()
	x = trans.location.x
	y = trans.location.y
	h = math.radians(trans.rotation.yaw)
	bb = vehicle.bounding_box
	h_l = bb.extent.x
	h_w = bb.extent.y
	return x, y, h, h_l, h_w
	
def get_pose_in_ego(pose_world, ego_pose_world):
	"""
	argument: pose in world coordinate
	return: pose in ego vehicle coordinate
	"""
	x_w,  y_w,  h_w  = pose_world
	ex_w, ey_w, eh_w = ego_pose_world
	R = np.array([[np.cos(eh_w), np.sin(eh_w)],
                 [-np.sin(eh_w), np.cos(eh_w)]])
	p_e = R.dot(np.array([x_w - ex_w, y_w - ey_w]))
	h_e = h_w - eh_w
	return p_e[0], p_e[1], h_e
	
def get_info_in_pixel(info_ego, d_behind, obs_range, img_size):
	"""
	argument: information in ego coordinate in meter, distance behind ego vehicle in meter, length of edge of FOV in meter, image size of edge of FOV in pixel
	return information in ego coordinate in pixel
	"""
	x_m, y_m, h_r, h_l_m, h_w_m = info_ego
	x_p = (x_m + d_behind) / obs_range * img_size
	y_p = y_m / obs_range * img_size + img_size / 2.0
	h_l_p = h_l_m / obs_range * img_size
	h_w_p = h_w_m / obs_range * img_size
	return x_p, y_p, h_r, h_l_p, h_w_p
	
def get_poly_corner(info):
	"""
	argument: tuple of x y h h_l h_w
	return: corner points representing a polygon
	"""
	x, y, h, h_l, h_w = info
	poly_no_heading = np.array([[h_l, h_w], [h_l, -h_w], [-h_l, -h_w], [-h_l, h_w]]).transpose()
	R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
	poly = np.matmul(R, poly_no_heading).transpose() + np.repeat([[x, y]], 4, axis=0)
	return poly
	
def get_pixels_inside_vehicle(info_pixel, grid_pixel):
	"""
	argument: vehicle info in pixel and pixel grid
	return: grid pixels that are contained by the vehicle
	"""
	poly = Path(get_poly_corner(info_pixel))
	grid = poly.contains_points(grid_pixel)
	isinPoly = np.where(grid == True)
	pixels = np.take(grid_pixel, isinPoly, axis=0)[0]
	return pixels
	
def is_in_distance_ahead(target_trans, reference_trans, distance):
	"""
	argument: target transform, reference transform, and allowed distance
	return: true if the target is in the range, false otherwise
	"""
	target_vector = np.array([target_trans.location.x - reference_trans.location.x, target_trans.location.y - reference_trans.location.y])
	if np.linalg.norm(target_vector) > distance:
		return False
	forward_vector = np.array([math.cos(math.radians(reference_trans.rotation.yaw)), math.sin(math.radians(reference_trans.rotation.yaw))])
	return math.acos(np.dot(forward_vector, target_vector) / np.linalg.norm(target_vector)) < np.pi/2.0
	
def create_carla_transform_from_pose(pose):
	"""
	argument: pose
	return: carla transform
	"""
	trans = carla.Transform()
	trans.location.x = pose[0]
	trans.location.y = pose[1]
	trans.rotation.yaw = pose[2]
	return trans
	
def display_to_rgb(display, obs_size):
	"""
	argument: pygame display
	return: rgb matrix in uint8
	"""
	rgb = np.fliplr(np.rot90(display, 3))  # flip to regular view
	rgb = skimage.transform.resize(rgb, (obs_size, obs_size))  # resize
	return rgb * 255
	
def rgb_to_display_surface(rgb, display_size):
	surface = pygame.Surface((display_size, display_size)).convert()
	display = skimage.transform.resize(rgb, (display_size, display_size))
	display = np.flip(display, axis=1)
	display = np.rot90(display, 1)
	pygame.surfarray.blit_array(surface, display)
	return surface
