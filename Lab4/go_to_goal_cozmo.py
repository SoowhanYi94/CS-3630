##Soowhan Yi

from skimage import color
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time
import sys
import asyncio
from PIL import Image
import math
from markers import detect, annotator
import time
from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *
from rrt import *
from cmap import *
    


#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom, self.grid)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)

def compute_odometry(curr_pose, cvt_inch=True):
    '''
    Compute the odometry given the current pose of the robot (use robot.pose)

    Input:
        - curr_pose: a cozmo.robot.Pose representing the robot's current location
        - cvt_inch: converts the odometry into grid units
    Returns:
        - 3-tuple (dx, dy, dh) representing the odometry
    '''

    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees
    
    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / grid.scale, dy / grid.scale

    return (dx, dy, diff_heading_deg(curr_h, last_h))


async def marker_processing(robot, camera_settings, show_diagnostic_image=False):
    '''
    Obtain the visible markers from the current frame from Cozmo's camera. 
    Since this is an async function, it must be called using await, for example:

        markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

    Input:
        - robot: cozmo.robot.Robot object
        - camera_settings: 3x3 matrix representing the camera calibration settings
        - show_diagnostic_image: if True, shows what the marker detector sees after processing
    Returns:
        - a list of detected markers, each being a 3-tuple (rx, ry, rh) 
          (as expected by the particle filter's measurement update)
        - a PIL Image of what Cozmo's camera sees with marker annotations
    '''

    global grid

    # Wait for the latest image from Cozmo
    image_event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # Convert the image to grayscale
    image = np.array(image_event.image)
    image = color.rgb2gray(image)
    
    # Detect the markers
    markers, diag = detect.detect_markers(image, camera_settings, include_diagnostics=True)

    # Measured marker list for the particle filter, scaled by the grid scale
    marker_list = [marker['xyh'] for marker in markers]
    marker_list = [(x/grid.scale, y/grid.scale, h) for x,y,h in marker_list]

    # Annotate the camera image with the markers
    if not show_diagnostic_image:
        annotated_image = image_event.image.resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(annotated_image, markers, scale=2)
    else:
        diag_image = color.gray2rgb(diag['filtered_image'])
        diag_image = Image.fromarray(np.uint8(diag_image * 255)).resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(diag_image, markers, scale=2)
        annotated_image = diag_image

    return marker_list, annotated_image


async def run(robot: cozmo.robot.Robot):

    global flag_odom_init, last_pose
    global grid, gui, pf
    global cmap, stopevent

    # start streaming
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    # Obtain the camera intrinsics matrix
    fx, fy = robot.camera.config.focal_length.x_y
    cx, cy = robot.camera.config.center.x_y
    camera_settings = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float)
            
    ###################

    # YOUR CODE HERE
    pickedUp= robot.is_picked_up
    localized = False
    goal = (6, 10, 90)
    is_at_goal = False
    pf = ParticleFilter(grid)
    flag = 5
    # Have the robot drive to the goal
    while flag > 0:
        if robot.is_picked_up:
            await robot.play_anim_trigger(cozmo.anim.Triggers.FrustratedByFailure).wait_for_completed()
            localized = False
            is_at_goal = False
            pf = ParticleFilter(grid)
        else :
            while not is_at_goal:
                # Make your code robust to the “kidnapped robot problem” by resetting your localization if the robot is picked up.
                if not robot.is_picked_up:
                    ## Determine the robot’s actions based on the current state of the localization system. 
                    if localized:
                        x, y, h, _ = compute_mean_pose(pf.particles)
                        if goal == (x, y, h):
                            await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabHappy).wait_for_completed()
                            is_at_goal = True
                        else:
                            dx = goal[0] - x
                            dy = goal[1] - y
                            h_temp = math.degrees(math.atan2(dy,dx))
                            dh = diff_heading_deg(h_temp, h)
                            dist = math.sqrt(dx ** 2 + dy ** 2) 
                            dh2 = diff_heading_deg(goal[2], h_temp)

                            if robot.is_picked_up:
                                localized = False
                                is_at_goal = False;
                                say('hi')
                                await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabHappy).wait_for_completed()
                            else:
                                global cmap
                                ########################################################################
                                # TODO: please enter your code below.
                                # Description of function provided in instructions
                                
                                #assume start position is in cmap and was loaded from emptygrid.json as [50, 35] already
                                #assume start angle is 0
                                #Add final position as goal point to cmap, with final position being defined as a point that is at midpoint of the map 
                                #you can get map width and map weight from cmap.get_size()
                                cozmo_pos = cmap.get_start() #get start position. This will be [50, 35] if we use emptygrid for actual robot
                                cozmo_angle = 0
                                map_width, map_height = cmap.get_size()
                                final_goal_center = Node((map_width*2/4, map_height*2/4))
                                cmap.add_goal(final_goal_center) #adding a goal first to be the center
                                print("center added as goal")
                                
                                #reset the current stored paths in cmap
                                #call the RRT function using your cmap as input, and RRT will update cmap with a new path to the target from the start position
                                #get path from the cmap
                                cmap.reset_paths()
                                RRT(cmap, cmap.get_start()) #get a path based on cmap target and start position
                                path = cmap.get_smooth_path() #smooth path
                                print("path created")   
                                
                                
                                #marked and update_cmap are both outputted from detect_cube_and_update_cmap(robot, marked, cozmo_pos).
                                #and marked is an input to the function, indicating which cubes are already marked
                                #So initialize "marked" to be an empty dictionary and "update_cmap" = False
                                marked = {} 
                                update_cmap = False
                                
                                #while the current cosmo position is not at the goal:
                                while cozmo_pos not in cmap.get_goals():
                                
                                    #break if path is none or empty, indicating no path was found
                                    if (path is None or len(path)==0):
                                        print("path is none") #sanmesh
                                        break
                                    
                                    # Get the next node from the path
                                    #drive the robot to next node in path. #First turn to the appropriate angle, and then move to it
                                    #you can calculate the angle to turn through a trigonometric function
                                    next_pos = path.pop(0)
                                    print('x: {0:.2f}, y: {1:.2f}'.format(next_pos.x, next_pos.y))
                                    angle = np.arctan2(next_pos.y - cozmo_pos.y, next_pos.x - cozmo_pos.x)
                                    print("driving robot to next node in path")
                                    if abs(angle - cozmo_angle) > 0.01:
                                        await robot.turn_in_place(cozmo.util.Angle(radians=angle - cozmo_angle)).wait_for_completed()
                                    await robot.drive_straight(cozmo.util.Distance(distance_mm=get_dist(cozmo_pos, next_pos)),
                                                         cozmo.util.Speed(speed_mmps=30)).wait_for_completed()
                                        
                                    # Update the current Cozmo position (cozmo_pos and cozmo_angle) to be new node position and angle 
                                    cozmo_pos = next_pos
                                    cozmo_angle = angle 
                                
                                    # Set new start position for replanning with RRT
                                    cmap.set_start(cozmo_pos)

                                    #detect any visible obstacle cubes and update cmap
                                    print("detect_cube_and_update_cmap")
                                    update_cmap, goal_center, marked = await detect_cube_and_update_cmap(robot, marked, cozmo_pos)
                                    
                                    #if we detected a cube, indicated by update_cmap, reset the cmap path, recalculate RRT, and get new paths 
                                    if update_cmap:
                                        # Found the goal
                                        cmap.reset_paths()
                                        RRT(cmap, cmap.get_start())
                                        path = cmap.get_smooth_path()

                    else:
                        pos = robot.pose
                        if robot.is_picked_up:
                            await robot.play_anim_trigger(cozmo.anim.Triggers.FrustratedByFailure).wait_for_completed()
                            localized = False
                            is_at_goal = False
                            pf = ParticleFilter(grid)
                        # Obtain odometry infomation
                        odometry = compute_odometry(pos)

                        # Obtain list of currently seen markers and their poses
                        markers, images = await marker_processing(robot, camera_settings)
                        # Update the particle filter using the above information
                        x, y, h, conf = pf.update(odometry, markers)

                        # Update the particle filter GUI (for debugging)
                        gui.show_particles(pf.particles)
                        gui.show_mean(x, y, h, conf)
                        gui.show_camera_image(images)
                        gui.updated.set()
                        last_pose = robot.pose
                        if conf:
                            localized = True
                        else:
                            await robot.turn_in_place(cozmo.util.degrees(10)).wait_for_completed()
                else:
                    await robot.play_anim_trigger(cozmo.anim.Triggers.FrustratedByFailure).wait_for_completed()
                    localized = False
                    is_at_goal = False
                    pf = ParticleFilter(grid)

            await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabHappy).wait_for_completed()
            flag -= 1

    ###################

class CozmoThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run,use_3d_viewer=False, use_viewer=False)


if __name__ == '__main__':
    global cmap, stopevent
    stopevent = threading.Event()
    cmap = CozMap("maps/emptygrid.json", node_generator)


    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()
    
    visualizer = Visualizer(cmap)
    visualizer.start()
    stopevent.set()

