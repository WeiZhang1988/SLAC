import numpy as np
import pygame
import time
import gym
from gym.spaces import Dict, Discrete, Box, Tuple
import carla

class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time
    
    def time(self):
        return self.timer()

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        try:
            self.display = pygame.display.set_mode(window_size, \
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        except Exception:
            print("display is not correctly created")
        
        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []
        
    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]
        
    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), \
        int(self.window_size[1]/self.grid_size[0])]
        
    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), \
        int(gridPos[0] * dis_size[1])]
    
    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)
    
    def get_sensor_list(self):
        return self.sensor_list
        
    def render(self):
        for s in self.sensor_list:
            if s.surface is not None:
                self.display.blit(self.get_display_offset(s.display_pos))
            
    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

class SensorManager:
    def __init__(self, world, sensor_type, transform, attached, \
    sensor_options, display_size, display_pos):
        self.surface = None
        self.gnss_msg = None
        self.imu_msg = None
        self.world = world
        self.sensor_type = sensor_type
        self.transform = transform
        self.attached = attached
        self.sensor_options = sensor_options
        self.display_size = display_size
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, \
        attached, sensor_options, display_size)
        self.timer = CustomTimer()
        
        self.time_processing = 0.0
        self.tics_processing = 0
        
    def init_sensor(self, sensor_type, transform, \
        attached, sensor_options, display_size):
        if sensor_type == 'RGBCamera':
            camera_bp = \
            self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', \
            str(self.display_size[0]))
            camera_bp.set_attribute('image_size_y', \
            str(self.display_size[1]))
            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])
            camera = self.world.spawn_actor(camera_bp, transform, \
            attach_to=attached)
            camera.listen(self.save_rgb_image)
            return camera
            
        elif sensor_type == 'LiDAR':
            lidar_bp = \
            self.world.get_blueprint_library().\
            find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('dropoff_general_rate', \
            lidar_bp.get_attribute('dropoff_general_rate').\
            recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', \
            lidar_bp.get_attribute('dropoff_intensity_limit').\
            recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', \
            lidar_bp.get_attribute('dropoff_zero_intensity').\
            recommended_values[0])
            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])
            lidar = self.world.spawn_actor(lidar_bp, transform, \
            attach_to=attached)
            lidar.listen(self.save_lidar_image)
            return lidar
            
        elif sensor_type == 'Radar':
            radar_bp = \
            self.world.get_blueprint_library().\
            find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])
            radar = self.world.spawn_actor(radar_bp, transform, \
            attach_to=attached)
            radar.listen(self.save_radar_image)
            return radar
            
        elif sensor_type == 'GNSS':
            gnss_bp = \
            self.world.get_blueprint_library().\
            find('sensor.other.gnss')
            for key in sensor_options:
                gnss_bp.set_attribute(key, sensor_options[key])
            gnss = self.world.spawn_actor(gnss_bp, transform, \
            attach_to=attached)
            gnss.listen(self.save_gnss_msg)
            return gnss
            
        elif sensor_type == 'IMU':
            imu_bp = \
            self.world.get_blueprint_library().\
            find('sensor.other.imu')
            for key in sensor_options:
                imu_bp.set_attribute(key, sensor_options[key])
            imu = self.world.spawn_actor(imu_bp, transform, \
            attach_to=attached)
            imu.listen(self.save_imu_msg)
            return imu
            
        else:
            return None
            
    def get_sensor(self):
        return self.sensor
        
    def save_rgb_image(self, image):
        t_start = self.timer.time()
        
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0,1))
        
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        
    def save_lidar_image(self, image):
        t_start = self.timer.time()
        
        lidar_range = 2.0 * float(self.sensor_options['range'])
        
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(self.display_size) / lidar_range
        lidar_data += (0.5 * self.display_size[0], \
        0.5 * self.display_size[1])
        lidar_data = np.fabs(lidar_data)
        lidar_data = lidar_data.astype(np.int32)
        lidar_img_size(self.display_size[0], self.display_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        
        self.surface = pygame.surfarray.make_surface(lidar_img)
        
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        
    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))
        
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        
    def save_gnss_msg(self, gnss_msg):
        lat = gnss_msg.latitude
        lon = gnss_msg.longitude
        alt = gnss_msg.altitude
        self.gnss_msg = (lat, lon, alt)
        
    def save_imu_msg(self, imu_msg):
        acc = (imu_msg.accelerometer.x, imu_msg.accelerometer.y, \
        imu_msg.accelerometer.z)
        gyro = (imu_msg.gyroscope.x, imu_msg.gyroscope.y, \
        imu_msg.gyroscope.z)
        cmpa = imu_msg.compass
        self.imu_msg = (acc, gyro, cmpa)

class CarlaEnv(gym.Env):
    def __init__(self, params):
        # parse parameters
        self.carla_port = params['carla_port']
        self.map_name = params['map_name']
        self.window_resolution = params['window_resolution']
        self.grid_size = params['grid_size']
        self.sync = params['sync']
        # connet to server
        self.client = carla.Client('localhost',self.carla_port)
        self.client.set_timeout(5.0)
        # get world and map
        self.world = self.client.get_world()
        self.world = self.client.load_world(self.map_name)
        self.map = self.world.get_map()
        print("Now we choose the map '{}'".format(self.map))
        self.original_settings = self.world.get_settings()
        if self.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = self.world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        
        self.display_manager = DisplayManager(self.grid_size, \
        self.window_resolution)
        self.display_size = self.display_manager.get_display_size()
        self.vehicle_list = []
        self.pedestrian_list = []
        self.sensor_list = []
        
        # acc and brake percentage, steering percentage, and reverse flag
        self.action_space = Tuple((Box([0.0 0.0 -1.0], 1.0, shape=(3,), \
        dtype=np.float32), Discrete(2)))
        # 4 cameras, 1 lidar, and 1 GNSS
        self.observation_space = Dict({
            'front_camera': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'left_camera': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'right_camera': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'rear_camera': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'lidar': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'gnss': Box(-np.inf, np.inf, shape=(3,), dtype=np.float32), 
            'imu': Tuple((\
            Box(-np.inf, np.inf, shape=(3,), dtype=np.float32), \
            Box(-np.inf, np.inf, shape=(3,), dtype=np.float32), \
            Box(-np.inf, np.inf, shape=(1,), dtype=np.float32), \
            )) 
        })
    def step(self,action=[0]):
        return 0,0,0,0
    def reset(self):
        for v in self.vehicle_list:
            if v.is_alive:
                v.destroy()
        for p in self.pedestrian_list:
            if p.is_alive:
                p.destroy()
        for s in self.sensor_list:
            if v.is_alive:
                v.destroy()
        ego_vehicle_bp = self.world.get_blueprint_library().\
        filter('charger_2020')[0]
        self.ego_vehicle = world.spawn_actor(ego_vehicle_bp, \
        random.choice(world.get_map().get_spawn_points()))
        self.vehicle_list.append(self.ego_vehicle)
        
        self.left_camera = SensorManager(self.world, 'RGBCamera', \
        carla.Transform(carla.Location(x=0, z=2.4), \
        carla.Rotation(yaw=-90)), self.ego_vehicle, {}, \
        self.display_size, display_pos=[0, 0])
        self.front_camera = SensorManager(self.world, 'RGBCamera', \
        carla.Transform(carla.Location(x=0, z=2.4), \
        carla.Rotation(yaw=+00)), self.ego_vehicle, {}, \
        self.display_size, display_pos=[0, 1])
        self.right_camera = SensorManager(self.world, 'RGBCamera', \
        carla.Transform(carla.Location(x=0, z=2.4), \
        carla.Rotation(yaw=+90)), self.ego_vehicle, {}, \
        self.display_size, display_pos=[0, 2])
        self.rear_camera = SensorManager(self.world, 'RGBCamera', \
        carla.Transform(carla.Location(x=0, z=2.4), \
        carla.Rotation(yaw=180)), self.ego_vehicle, {}, \
        self.display_size, display_pos=[1, 1])
        
        self.lidar = SensorManager(self.world, 'LiDAR', \
        carla.Transform(carla.Location(x=0, z=2.4)), self.ego_vehicle, \
        {'channels' : '64', 'range' : '100',  \
        'points_per_second': '250000', 'rotation_frequency': '20'}, \
        self.display_size, display_pos=[1, 0])
    def close(self):
        pass