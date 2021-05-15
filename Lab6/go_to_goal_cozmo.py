##Soowhan Yi
## https://youtu.be/Z4BTw_0Em9o

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
from gui_rrt import *
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *
from utils_rrt import *
from cmap import *
MAX_NODES = 20000

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

async def cube_pick(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    cube = await robot.world.wait_for_observed_light_cube(timeout=10)
    robot.pickup_object(cube, num_retries=2).wait_for_completed()
    time.sleep(2)

#async def 
def calculate_dist_dh(x,y,h,goal):
    dx = goal[0] - x
    dy = goal[1] - y
    h_temp = math.degrees(math.atan2(dy,dx))
    dh = diff_heading_deg(h_temp, h)
    dist = math.sqrt(dx ** 2 + dy ** 2) 
    dh2 = diff_heading_deg(goal[2], h_temp)
    return dist, dh, dh2

async def run(robot: cozmo.robot.Robot):

    global flag_odom_init, last_pose
    global grid, gui, pf
    global cmap, stopevent, hasCube, cozmo_angle

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
    cozmo_angle = 0
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
                        if not is_at_goal:
                            dist, dh, dh2 = calculate_dist_dh(x,y,h,goal)
                            if robot.is_picked_up:
                                localized = False
                                is_at_goal = False;
                                say('hi')
                                await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabHappy).wait_for_completed()
                            else:
                                time.sleep(2)
                                await robot.turn_in_place(cozmo.util.degrees(dh)).wait_for_completed()
                                time.sleep(2)
                                await robot.drive_straight(cozmo.util.distance_inches(dist), cozmo.util.speed_mmps(40)).wait_for_completed()
                                time.sleep(2)
                                await robot.turn_in_place(cozmo.util.degrees(dh2)).wait_for_completed()
                                time.sleep(2)
                                robot.camera.image_stream_enabled = True
                                robot.camera.color_image_enabled = False
                                robot.camera.enable_auto_exposure()
                                cube = await robot.world.wait_for_observed_light_cube(timeout=10)
                                await robot.pickup_object(cube, num_retries=2).wait_for_completed()
                                await robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
                                is_at_goal = True
                                hasCube = True
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
                        # gui.show_particles(pf.particles)
                        # gui.show_mean(x, y, h, conf)
                        # gui.show_camera_image(images)
                        # gui.updated.set()
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
            time.sleep(2)
            await CozmoPlanning(robot)
            time.sleep(5)
            if hasCube:
                await robot.place_object_on_ground_here(cube).wait_for_completed()
                hasCube = False

            else:
                #await robot.turn_in_place(cozmo.util.degrees(90-cozmo_angle)).wait_for_completed()
                cube = await robot.world.wait_for_observed_light_cube(timeout=10)
                await robot.pickup_object(cube, num_retries=2).wait_for_completed()
                hasCube = True
                #await robot.turn_in_place(cozmo.util.degrees(90)).wait_for_completed()
            await robot.turn_in_place(cozmo.util.degrees(-cozmo_angle)).wait_for_completed()
            # hasCube = False
            # await CozmoPlanning(robot)
            # await robot.play_anim_trigger(cozmo.anim.Triggers.FrustratedByFailure).wait_for_completed()
            # time.sleep(2)
            # cube = await robot.world.wait_for_observed_light_cube(timeout=10)
            # await robot.pickup_object(cube, num_retries=2).wait_for_completed()
            # await robot.turn_in_place(cozmo.util.degrees(-90)).wait_for_completed()
            #await CozmoPlanning(robot)
            flag -= 1

    ###################
    
    

def step_from_to(node0, node1, limit=75):
    ########################################################################
    # TODO: please enter your code below.
    # 1. If distance between two nodes is less than limit, return node1
    # 2. Otherwise, return a node in the direction from node0 to node1 whose
    #    distance to node0 is limit. Recall that each iteration we can move
    #    limit units at most
    # 3. Hint: please consider using np.arctan2 function to get vector angle
    # 4. Note: remember always return a Node object
    ############################################################################
    
    #############################################################################
    # Instructors Solution
    if get_dist(node0, node1) < limit:
        return node1
    else:
        theta = np.arctan2(node1.y - node0.y, node1.x - node0.x)
        return Node((node0.x + limit * np.cos(theta), node0.y + limit * np.sin(theta)))


def node_generator(cmap):
    rand_node = None
    ############################################################################
    # TODO: please enter your code below.
    # 1. Use CozMap width and height to get a uniformly distributed random node
    # 2. Use CozMap.is_inbound and CozMap.is_inside_obstacles to determine the
    #    legitimacy of the random node.
    # 3. Note: remember always return a Node object
    ############################################################################

    #############################################################################
    # Instructors Solution
    if np.random.rand() < 0.05:
    #if np.random.rand() < 0.00:
        return Node((cmap.get_goals()[0].x, cmap.get_goals()[0].y))

    else:
        while True:
            rand_node = Node((np.random.uniform(cmap.width),\
                     np.random.uniform(cmap.height)))
            if cmap.is_inbound(rand_node) \
                    and (not cmap.is_inside_obstacles(rand_node)):
                break
        return rand_node
    ############################################################################


def RRT(cmap, start):
    # cmap.add_node(start)
    # map_width, map_height = cmap.get_size()
    # while (cmap.get_num_nodes() < MAX_NODES):
    #     ########################################################################
    #     # TODO: please enter your code below.
    #     # 1. Use CozMap.get_random_valid_node() to get a random node. This
    #     #    function will internally call the node_generator above
    #     # 2. Get the nearest node to the random node from RRT
    #     # 3. Limit the distance RRT can move
    #     # 4. Add one path from nearest node to random node
    #     #
    #     rand_node = None
    #     nearest_node = None
    #     pass
    #     ########################################################################
    #     time.sleep(0.01)
    #     cmap.add_path(nearest_node, rand_node)
    #     if cmap.is_solved():
    #         break

    # path = cmap.get_path()
    # smoothed_path = cmap.get_smooth_path()

    # if cmap.is_solution_valid():
    #     print("A valid solution has been found :-) ")
    #     print("Nodes created: ", cmap.get_num_nodes())
    #     print("Path length: ", len(path))
    #     print("Smoothed path length: ", len(smoothed_path))
    # else:
    #     print("Please try again :-(")
    
    ############################################################################
    # instructors solution
    cmap.add_node(start)
    map_width, map_height = cmap.get_size()
    while (cmap.get_num_nodes() < MAX_NODES):
        rand_node = node_generator(cmap)
        #rand_node = cmap.get_random_valid_node()
        nearest_node_dist = np.sqrt(map_height ** 2 + map_width ** 2)
        nearest_node = None
        for node in cmap.get_nodes():
            if get_dist(node, rand_node) < nearest_node_dist:
                nearest_node_dist = get_dist(node, rand_node)
                nearest_node = node
        rand_node = step_from_to(nearest_node, rand_node)
        time.sleep(0.01)
        cmap.add_path(nearest_node, rand_node)
        if cmap.is_solved():
            break

    path = cmap.get_path()
    smoothed_path = cmap.get_smooth_path()

    if cmap.is_solution_valid():
        print("A valid solution has been found :-) ")
        print("Nodes created: ", cmap.get_num_nodes())
        print("Path length: ", len(path))
        print("Smoothed path length: ", len(smoothed_path))
    else:
        print("Please try again :-(")


async def CozmoPlanning(robot: cozmo.robot.Robot):
    # Allows access to map and stopevent, which can be used to see if the GUI
    # has been closed by checking stopevent.is_set()
    global cmap, stopevent, hasCube, cozmo_angle
    
    if hasCube:
        cmap = CozMap("maps/emptygrid.json", node_generator)
        cmap.add_goal(Node((475, 400)))
    else:
        cmap = CozMap("maps/emptygrid2.json", node_generator)
        cmap.add_goal(Node((152, 254)))

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
    final_goal_center = cmap.get_goals()
     #adding a goal first to be the center
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
def get_global_node(local_angle, local_origin, node):
    """Helper function: Transform the node's position (x,y) from local coordinate frame specified by local_origin and local_angle to global coordinate frame.
                        This function is used in detect_cube_and_update_cmap()
        Arguments:
        local_angle, local_origin -- specify local coordinate frame's origin in global coordinate frame
        local_angle -- a single angle value
        local_origin -- a Node object

        Outputs:
        new_node -- a Node object that decribes the node's position in global coordinate frame
    """
    ########################################################################
    # TODO: please enter your code below.
    local_vec = np.array([[node.x], [node.y], [1]])
    global_T_local = np.array([[np.cos(local_angle), -np.sin(local_angle), local_origin.x],
                               [np.sin(local_angle), np.cos(local_angle), local_origin.y],
                               [0, 0, 1]])
    global_vec = global_T_local.dot(local_vec)
    return Node((int(global_vec[0]), int(global_vec[1])))


async def detect_cube_and_update_cmap(robot, marked, cozmo_pos):
    """Helper function used to detect obstacle cubes and the goal cube.
       1. When a valid goal cube is detected, old goals in cmap will be cleared and a new goal corresponding to the approach position of the cube will be added.
       2. Approach position is used because we don't want the robot to drive to the center position of the goal cube.
       3. The center position of the goal cube will be returned as goal_center.

        Arguments:
        robot -- provides the robot's pose in G_Robot
                 robot.pose is the robot's pose in the global coordinate frame that the robot initialized (G_Robot)
                 also provides light cubes
        cozmo_pose -- provides the robot's pose in G_Arena
                 cozmo_pose is the robot's pose in the global coordinate we created (G_Arena)
        marked -- a dictionary of detected and tracked cubes (goal cube not valid will not be added to this list)

        Outputs:
        update_cmap -- when a new obstacle or a new valid goal is detected, update_cmap will set to True
        goal_center -- when a new valid goal is added, the center of the goal cube will be returned
    """
    global cmap

    # Padding of objects and the robot for C-Space
    cube_padding = 40.
    cozmo_padding = 100.

    # Flags
    update_cmap = False
    goal_center = None

    # Time for the robot to detect visible cubes
    time.sleep(1)

    for obj in robot.world.visible_objects:

        if obj.object_id in marked:
            continue

        # Calculate the object pose in G_Arena
        # obj.pose is the object's pose in G_Robot
        # We need the object's pose in G_Arena (object_pos, object_angle)
        dx = obj.pose.position.x - robot.pose.position.x
        dy = obj.pose.position.y - robot.pose.position.y

        object_pos = Node((cozmo_pos.x+dx, cozmo_pos.y+dy))
        object_angle = obj.pose.rotation.angle_z.radians

        # Define an obstacle by its four corners in clockwise order
        obstacle_nodes = []
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, -cube_padding))))
        obstacle_nodes.append(get_global_node(object_angle, object_pos, Node((-cube_padding, cube_padding))))
        cmap.add_obstacle(obstacle_nodes)
        marked[obj.object_id] = obj
        update_cmap = True

    return update_cmap, goal_center, marked


class CozmoThread(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run,use_3d_viewer=False, use_viewer=False)
        #cozmo.run_program(CozmoPlanning, hasCube = False,use_3d_viewer=False, use_viewer=False)
        #stopevent.set()


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()
    stopevent.set()

