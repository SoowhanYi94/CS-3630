from grid import CozGrid
from particle import Particle
from utils import grid_distance, rotate_point, diff_heading_deg, add_odometry_noise, add_marker_measurement_noise
import setting
import math
import numpy as np


def motion_update(particles, odom, grid):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*
        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used for boundary checking
        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []
    for p in particles:
        x, y, h = p.xyh
        dx, dy, dh = odom
        dx, dy = rotate_point(dx, dy, h)
        odom_temp = (x + dx, y + dy, h + dh)
        x, y, h = add_odometry_noise(odom_temp, setting.ODOM_HEAD_SIGMA, setting.ODOM_TRANS_SIGMA)
        motion_particles.append(Particle(x, y, h))
    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before measurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    if len(measured_marker_list) == 0 or len(particles) == 0:
        return particles
    weights = []
    particleLength = len(particles)
    num = 0
    for p in particles:
        probability = get_weight(p, measured_marker_list, grid)
        weights.append(probability)
    if sum(weights) == 0:
        weights = [1.0/len(weights) for i in weights] 
    else:
        weights = np.divide(weights, np.sum(weights))
    measured_particles = np.random.choice(particles, \
        size = particleLength - 300, replace = True, p = weights).tolist() + Particle.create_random(count=300, grid=grid)

    return measured_particles

def get_weight(particle, measured_marker_list, grid):
    x, y, h = particle.xyh
    probability = 1.0
    weight = 0
    sim_marker_list = particle.read_markers(grid)
    if len(sim_marker_list) == 0:
        return 0
    if grid.is_free(x,y):
        for measured_marker in measured_marker_list:
            max_prob = -1.0
            pair = None
            measured_marker = add_marker_measurement_noise(measured_marker, setting.MARKER_TRANS_SIGMA, setting.MARKER_ROT_SIGMA)
            for sim_marker in sim_marker_list:
                dist = grid_distance(measured_marker[0], measured_marker[1], sim_marker[0], sim_marker[1])
                angle = diff_heading_deg(measured_marker[2], sim_marker[2])
                prob_temp = np.exp(-(dist**2)/(2 * setting.MARKER_TRANS_SIGMA**2) -(angle**2)/(2 * setting.MARKER_ROT_SIGMA**2))
                if prob_temp > max_prob:
                    max_prob = prob_temp
                    pair = sim_marker
            if pair != None:
                probability *= max_prob
        weight = probability
    return weight
