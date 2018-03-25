import random
import math
import numpy as np
import sys
import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import draw

width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

show_sensors = True
draw_screen = True

class GameState:
    def __init__(self):
        self.crashed = False

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        self.create_ob(100, 100, 0.5)

        self.num_steps = 0
        self.cnt = 0
        self.mx = 0
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 6),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 6),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 6),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 6)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['grey7']
        self.space.add(static)

        self.ast = []
        self.ast.append(self.create_ast(200, 350, 0.5))
        self.ast.append(self.create_ast(700, 200, 1))
        self.ast.append(self.create_ast(600, 600, 0.8))
        self.ast.append(self.create_ast(100, 600, 0.4))
        self.create_mover()
    '''def create_ast(self, x, y, r):
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body'''

    def create_ast(self, x, y, r):
     #   c_body = pymunk.Body(mass=0, moment=0, \
      #      body_type= <class'CP_BODY_TYPE_DYNAMIC'>)
        c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        vertices = [(40*r, 0), (80*r, 0), (120*r, 40*r), (120*r, 80*r), (80*r, 120*r),
         (40*r, 120*r), (0, 80*r), (0, 40*r)]
        c_body.position = x, y
        c_shape = pymunk.Poly(c_body, vertices, radius = 20)
       # c_shape = pymunk.Poly(c_body, [(0.0, -30.0), (19.0, -23.0), (30.0, -5.0), (26.0, 15.0), (10.0, 28.0), (-10.0, 28.0), (-26.0, 15.0), (-30.0, -5.0), (-19.0, -23.0)])
        c_shape.color = pygame.color.THECOLORS["gainsboro"]
        c_shape.elasticity = 1.0
        self.space.add(c_body, c_shape)
        return c_body

    def create_mover(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.mover_body = pymunk.Body(1, inertia)
        self.mover_body.position = 50, height - 100
        self.mover_shape = pymunk.Circle(self.mover_body, 30)
        self.mover_shape.color = THECOLORS["lightgrey"]
        self.mover_shape.elasticity = 1.0
        self.mover_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.mover_body.angle)
        self.space.add(self.mover_body, self.mover_shape)

    def create_ob(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.ob_body = pymunk.Body(1, inertia)
        self.ob_body.position = x, y
        self.ob_shape = pymunk.Circle(self.ob_body, 25)
        self.ob_shape.color = THECOLORS["green"]
        self.ob_shape.elasticity = 1.0
        self.ob_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.ob_body.angle)
        self.ob_body.apply_impulse(driving_direction)
        self.space.add(self.ob_body, self.ob_shape)

    def frame_step(self, action):
        if action == 0:  # Turn left.
            self.ob_body.angle -= .2
        elif action == 1:  # Turn right.
            self.ob_body.angle += .2
        # Move ast.
        if self.num_steps % 100 == 0:
            self.move_ast()
            #print("WITh")

        # Move mover.
        if self.num_steps % 5 == 0:
            self.move_mover()

        font = pygame.font.Font(None, 50)
        if self.cnt > self.mx:
            self.mx = self.cnt
        driving_direction = Vec2d(1, 0).rotated(self.ob_body.angle)
        self.ob_body.velocity = 100 * driving_direction
        CURSOR_UP_ONE = '\x1b[1A' 
        ERASE_LINE = '\x1b[2K'
        screen.fill(THECOLORS["black"])
        #print("Caught")
        draw(screen, self.space)
        screen.blit(font.render("Score: " + str(self.cnt), 
            1, THECOLORS["white"]), (400,0))
        screen.blit(font.render("MAX Score: " + str(self.mx), 
            1, THECOLORS["white"]), (600,0))
        #pygame.display.set_caption("Game")
        #for _ in range(3):
         #   sys.stdout.write(CURSOR_UP_ONE)
          #  sys.stdout.write(ERASE_LINE)
        self.space.step(1./10) # frame change................................
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        # Get the current lomoverion and the readings there.
        x, y = self.ob_body.position
        readings = self.get_sensor_readings(x, y, self.ob_body.angle)
        normalized_readings = [(x-20.0)/20.0 for x in readings] 
        state = np.array([normalized_readings])

        # Set the reward.
        # ob crashed when any reading == 1
        if self.ob_is_crashed(readings):
            self.crashed = True
            reward = -500
            self.cnt = 0
            self.recover_from_crash(driving_direction)
        else:
            # Higher readings are better, so return the sum.
            reward = -5 + int(self.sum_readings(readings) / 10)
        self.num_steps += 1
        self.cnt += 1

        return reward, state

    def move_ast(self):
        # Randomly move ast around.
        for ast in self.ast:
            speed = random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.ob_body.angle + random.randint(-2, 2))
            ast.velocity = speed * direction

    def move_mover(self):
        speed = random.randint(20, 200)
        self.mover_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.mover_body.angle)
        self.mover_body.velocity = speed * direction

    def ob_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        print("CRASH")
        while self.crashed:
            self.ob_body.velocity = -100 * driving_direction
            self.crashed = False
            for i in range(10):
                self.ob_body.angle += .2 
                screen.fill(THECOLORS["orange"])  
                draw(screen, self.space)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()

    def sum_readings(self, readings):
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sensor_readings(self, x, y, angle):
        readings = []
        arm_left = self.make_sensor_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))
        #arm = self.make_sonar(x, y)
        #g = self.draw_sensor(arm, x, y, angle, 0)
        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0
        cntr = 0
        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                if self.cnt % 2:
                    if cntr % 2:
                        pygame.draw.circle(screen, (255, 0, 0), (rotated_p), 2)
                    else:
                        pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)
                else:
                    if cntr % 2:
                        pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)
                    else:
                        pygame.draw.circle(screen, (255, 0, 0), (rotated_p), 2)
            cntr += 1
        # Return the distance for the arm.
        return i

    def make_sensor_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def make_sonar(self, x, y):
        spread = 10
        distance = 20
        arm_points = []
        for i in range (1, 10):
            arm_points.append((distance + x + (spread * i) + i*50, y))

        return arm_points

    def draw_sensor(self, arm, x, y, angle, offset):
        i = 0
        cntr = 0
        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x + i, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                if self.cnt % 2:
                    if cntr % 2:
                        pygame.draw.circle(screen, (255, 0, 0), (rotated_p), 2)
                    else:
                        pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)
                else:
                    if cntr % 2:
                        pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)
                    else:
                        pygame.draw.circle(screen, (255, 0, 0), (rotated_p), 2)
            cntr += 1
        # Return the distance for the arm.
        return i

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['black']:
            return 0
        else:
            return 1

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))
