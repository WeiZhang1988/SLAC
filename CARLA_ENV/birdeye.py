import carla
import math
import pygame
import weakref

# Colors
COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)


COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

class Util(object):
	@staticmethod
	def blits(destination_surface, source_surfaces, rect=None, blend_mode=0):
		for surface in source_surfaces:
			destination_surface.blit(surface[0], surface[1], rect, blend_mode)
	
	@staticmethod
	def length(v):
		return math.sqrt(v.x**2 + v.y**2 + v.z**2)
	
	@staticmethod
	def get_bounding_box(actor):
		bb = actor.trigger_volume.extent
		corners = [carla.Location(x=-bb.x, y=-bb.y),
			carla.Location(x=bb.x, y=-bb.y),
			carla.Location(x=bb.x, y=bb.y),
			carla.Location(x=-bb.x, y=bb.y),
			carla.Location(x=-bb.x, y=-bb.y)]
		corners = [x + actor.trigger_volume.location for x in corners]
		t = actor.get_transform()
		t.transform(corners)
		return corners
		
class MapImage(object):
	def __init__(self, carla_world, carla_map, pixels_per_meter):
		self.pixels_per_meter = pixels_per_meter
		self.scale = 1.0
		
		waypoints = carla_map.generate_waypoints(2)
		margin = 50
		max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
		max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
		min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
		min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin
		
		self.width = max(max_x - min_x, max_y - min_y)
		self.world_offset = (min_x, min_y)
		
		width_in_pixels = int(self.pixels_per_meter * self.width)
		
		self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
		
	def draw_road_map(self, map_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
		map_surface.fill(COLOR_BLACK)
		precision = 0.05
		
		def lane_marking_color_to_tango(lane_marking_color):
			tango_color = COLOR_BLACK
			
			if lane_marking_color == carla.LaneMarkingColor.White:
				tango_color = COLOR_ALUMINIUM_2
			
			elif lane_marking_color == carla.LaneMarkingColor.Blue:
				tango_color = COLOR_SKY_BLUE_0
			
			elif lane_marking_color == carla.LaneMarkingColor.Green:
				tango_color = COLOR_CHAMELEON_0
			
			elif lane_marking_color == carla.LaneMarkingColor.Red:
				tango_color = COLOR_SCARLET_RED_0
			
			elif lane_marking_color == carla.LaneMarkingColor.Yellow:
				tango_color = COLOR_ORANGE_0
			
			return tango_color
		
		def draw_solid_line(surface, color, closed, points, width):
			if len(points) >= 2:
				pygame.draw.lines(surface, color, closed, points, width)
		
		def draw_broken_line(surface, color, closed, points, width):
			broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]
			for line in broken_lines:
				pygame.draw.lines(surface, color, closed, line, width)
		
		def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign):
			margin = 0.25
			marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
			if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):
				return [(lane_marking_type, lane_marking_color, marking_1)]
			else:
				marking_2 = [world_to_pixel(lateral_shift(w.transform, sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]
				if lane_marking_type == carla.LaneMarkingType.SolidBroken:
					return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1), (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
				elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
					return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1), (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
				elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
					return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1), (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
				elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
					return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1), (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
			return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]

		def draw_lane(surface, lane, color):
			for side in lane:
				lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
				lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]
				polygon = lane_left_side + [x for x in reversed(lane_right_side)]
				polygon = [world_to_pixel(x) for x in polygon]
				
				if len(polygon) > 2:
					pygame.draw.polygon(surface, color, polygon, 5)
					pygame.draw.polygon(surface, color, polygon)

		def draw_lane_marking(surface, waypoints):
			# Left Side
			draw_lane_marking_single_side(surface, waypoints[0], -1)
			# Right Side
			draw_lane_marking_single_side(surface, waypoints[1], 1)

		def draw_lane_marking_single_side(surface, waypoints, sign):
			lane_marking = None
			
			marking_type = carla.LaneMarkingType.NONE
			previous_marking_type = carla.LaneMarkingType.NONE
			
			marking_color = carla.LaneMarkingColor.Other
			previous_marking_color = carla.LaneMarkingColor.Other
			
			markings_list = []
			temp_waypoints = []
			current_lane_marking = carla.LaneMarkingType.NONE
			for sample in waypoints:
				lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking
				
				if lane_marking is None:
					continue
				
				marking_type = lane_marking.type
				marking_color = lane_marking.color
				
				if current_lane_marking != marking_type:
					markings = get_lane_markings(previous_marking_type, lane_marking_color_to_tango(previous_marking_color), temp_waypoints, sign)
					current_lane_marking = marking_type
					
					for marking in markings:
						markings_list.append(marking)
					
					temp_waypoints = temp_waypoints[-1:]
					
				else:
					temp_waypoints.append((sample))
					previous_marking_type = marking_type
					previous_marking_color = marking_color
			
			last_markings = get_lane_markings(previous_marking_type, lane_marking_color_to_tango(previous_marking_color), temp_waypoints, sign)
			# Add last marking
			for marking in last_markings:
				markings_list.append(marking)
				
			for markings in markings_list
				if markings[0] == carla.LaneMarkingType.Solid:
					draw_solid_line(surface, markings[1], False, markings[2], 2)
				elif markings[0] == carla.LaneMarkingType.Broken:
					draw_broken_line(surface, markings[1], False, markings[2], 2)
			
		def draw_traffic_signs(surface, font_surface, actor, color=COLOR_ALUMINIUM_2, trigger_color=COLOR_PLUM_0):
			transform = actor.get_transform()
			waypoint = carla_map.get_waypoint(transform.location)
			
			angle = -waypoint.transform.rotation.yaw - 90.0
			font_surface = pygame.transform.rotate(font_surface, angle)
			pixel_pos = world_to_pixel(waypoint.transform.location)
			offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
			surface.blit(font_surface, offset)
			
			# Draw line in front of stop
			forward_vector = carla.Location(waypoint.transform.get_forward_vector())
			left_vector = carla.Location(-forward_vector.y, forward_vector.x, forward_vector.z) * waypoint.lane_width / 2.0 * 0.7
			line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)), (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]
			line_pixel = [world_to_pixel(p) for p in line]
			pygame.draw.lines(surface, color, True, line_pixel, 2)
		
		def lateral_shift(transform, shift):
			transform.rotation.yaw += 90
			return transform.location + shift * transform.get_forward_vector()
			
		def draw_topology(carla_topology, index):
			topology = [x[index] for x in carla_topology]
			topology = sorted(topology, key=lambda w: w.transform.location.z)
			set_waypoints = []
			for waypoint in topology:
				waypoints = [waypoint]
				
				nxt = waypoint.next(precision)
				if len(nxt) > 0:
					nxt = nxt[0]
					while nxt.road_id == waypoint.road_id:
						waypoints.append(nxt)
						nxt = nxt.next(precision)
						if len(nxt) > 0:
							nxt = nxt[0]
						else:
							break
				set_waypoints.append(waypoints)
				
				# Draw Shoulders, Parkings and Sidewalks
				PARKING_COLOR = COLOR_ALUMINIUM_4_5
				SHOULDER_COLOR = COLOR_ALUMINIUM_5
				SIDEWALK_COLOR = COLOR_ALUMINIUM_3
				
				shoulder = [[], []]
				parking = [[], []]
				sidewalk = [[], []]
				
				for w in waypoints:
					l = w.get_left_lane()
					while l and l.lane_type != carla.LaneType.Driving:
						if l.lane_type == carla.LaneType.Shoulder:
							shoulder[0].append(l)
						
						if l.lane_type == carla.LaneType.Parking:
							parking[0].append(l)
						
						if l.lane_type == carla.LaneType.Sidewalk:
							sidewalk[0].append(l)
							
						l = l.get_left_lane()
						
					r = w.get_right_lane()
					while r and r.lane_type != carla.LaneType.Driving:
						if r.lane_type == carla.LaneType.Shoulder:
							shoulder[1].append(r)
						
						if r.lane_type == carla.LaneType.Parking:
							parking[1].append(r)
							
						if r.lane_type == carla.LaneType.Sidewalk:
							sidewalk[1].append(r)
						
						r = r.get_right_lane()
						
				draw_lane(map_surface, shoulder, SHOULDER_COLOR)
				draw_lane(map_surface, parking, PARKING_COLOR)
				draw_lane(map_surface, sidewalk, SIDEWALK_COLOR)
				
			# Draw Roads
			for waypoints in set_waypoints:
				waypoint = waypoints[0]
				road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
				road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
				
				polygon = road_left_side + [x for x in reversed(road_right_side)]
				polygon = [world_to_pixel(x) for x in polygon]
				
				if len(polygon) > 2:
					pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
					pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)
					
				# Draw Lane Markings
				if not waypoint.is_junction:
					draw_lane_marking(map_surface, [waypoints, waypoints])
					
		topology = carla_map.get_topology()
		draw_topology(topology, 0)
		
		actors = carla_world.get_actors()
		
		# Draw Traffic Signs
		font_size = world_to_pixel_width(1)
		font = pygame.font.SysFont('Arial', font_size, True)
		
		stops = [actor for actor in actors if 'stop' in actor.type_id]
		yields = [actor for actor in actors if 'yield' in actor.type_id]
		
		stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
		stop_font_surface = pygame.transform.scale(stop_font_surface, (stop_font_surface.get_width(), stop_font_surface.get_height() * 2))
		
		yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
		yield_font_surface = pygame.transform.scale(yield_font_surface, (yield_font_surface.get_width(), yield_font_surface.get_height() * 2))
		
		for ts_stop in stops:
			draw_traffic_signs(map_surface, stop_font_surface, ts_stop, trigger_color=COLOR_SCARLET_RED_1)
			
		for ts_yield in yields:
			draw_traffic_signs(map_surface, yield_font_surface, ts_yield, trigger_color=COLOR_ORANGE_1)
			
	def world_to_pixel(self, location, offset=(0,0)):
		x = self.scale * self.pixels_per_meter * (location.x - self.world_offset[0])
		y = self.scale * self.pixels_per_meter * (location.y - self.world_offset[1])
		return [int(x - offset[0]), int(y - offset[1])]
		
	def world_to_pixel_width(self, width):
		return int(self.scale * self.pixels_per_meter * width)
		
class BirdeyeView(object):
	def __init__(self, world, pixel_per_meter, screen_size, pixel_ahead_vehicle):
		self.pixel_per_meter = pixel_per_meter
		self.screen_size = screen_size
		self.pixel_ahead_vehicle = pixel_ahead_vehicle
		
		self.world = world
		self.town_map = self.world.get_map()
		self.actors_with_transforms = []
		
		self.hero_actor = None
		self.hero_id = None
		self.hero_transform = None
		
		self.vehicle_polygons = []
		self.walker_polygons = []
		
		self.map_image = MapImage(carla_world=self.world, carla_map=self.town_map,pixel_per_meter=self.pixel_per_meter)
		
		self.map_surface_size = self.map_image.big_map_surface.get_width()
		
		self.actors_surface = pygame.Surface((self.map_image.big_map_surface.get_width(), self.map_image.big_map_surface.get_height()))
		self.actors_surface.set_colorkey(COLOR_BLACK)
		
		self.hero_surface = pygame.Surface((self.map_surface_size,self.map_surface_size)).convert()
		
		self.result_surface = pygame.Surface((self.map_surface_size,self.map_surface_size)).convert()
		self.result_surface.set_colorkey(COLOR_BLACK)
	
	def set_hero(self, hero_actor, hero_id):
		self.hero_actor = hero_actor
		self.hero_id = hero_id
		
	def tick(self):
		actors = self.world.get_actors()
		self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
		if self.hero_actor is not None:
			self.hero_transform = self.hero_actor.get_transform()
	
	def render_hist_actors(self, surface, actor_polygons, actor_type, world_to_pixel, num):
		lp = len(actor_polygons)
		color = COLOR_SKY_BLUE_0
		
		for i in range(max(0,lp-num),lp):
			for ID, poly in actor_polygons[i].items():
				corners = []
				for p in poly:
					corners.append(carla.Location(x=p[0],y=p[1]))
				corners.append(carla.Location(x=poly[0][0],y=poly[0][1]))
				corners = [world_to_pixel(p) for p in corners]
				color_value = max(0.8 - 0.8/lp*(i+1), 0)
				if ID == self.hero_id:
					color = pygame.Color(0,0,255)
				else:
					if actor_type == 'vehicle':
						color = pygame.Color(255,0,0)
					elif actor_type == 'walker':
						color = pygame.Color(255,255,0)
				pygame.draw.polygon(surface, color, corners)
				
	def render_actors(self, surface, vehicles, walkers):
		self.render_hist_actors(surface, vehicles, 'vehicle', self.map_image.world_to_pixel, 10)
		self.render_hist_actors(surface, walkers, 'walker', self.map_image.world_to_pixel, 10)
		
	def render(self, display):
		self.tick()
		
		if self.actors_with_transforms is None:
			return
		self.result_surface.fill(COLOR_BLACK)
		
		self.actors_surface.fill(COLOR_BLACK)
		self.render_actors(self.actors_surface, self.vehicle_polygons, self.walker_polygons)

		surfaces = [(self.map_image.surface, (0, 0)),(self.actors_surface, (0, 0))]
		
		angle = 0.0 if self.hero_actor is None else self.hero_transform.rotation.yaw + 90.0
		
		if self.hero_actor is not None:
			hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)
			hero_front = self.hero_transform.get_forward_vector()
			translation_offset = (hero_location_screen[0] - self.hero_surface.get_width() / 2.0 + hero_front.x * self.pixel_ahead_vehicle, hero_location_screen[1] - self.hero_surface.get_height() / 2.0 + hero_front.y * self.pixel_ahead_vehicle)
			
			Util.blits(self.result_surface, surfaces)
