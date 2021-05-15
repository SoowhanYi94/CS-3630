import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
from imgclassification import ImageClassifier as IC
import numpy
import sklearn 
import joblib
from skimage import io, feature, filters, exposure, color, measure
import sys
def run(robot: cozmo.robot.Robot):
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()

def FSM(robot: cozmo.robot.Robot, label):
    if label == 'drone':
        drone(robot)
    elif label == 'order':
        order(robot)
    elif label == 'inspection': 
        inspection(robot)
    idle(robot)

def idle(robot: cozmo.robot.Robot):

    # Look out for the secret symbols by monitoring the stream of images from the camera.
    # Classify each symbol (image) using the model you developed in Lab1. If one of the symbols is
    # recognized (i.e. not “none”), use the built-in text-to-speech functionality to have the robot say the
    # name of the recognized symbol, then switch to the appropriate state 
    robot.set_head_angle(degrees(0)).wait_for_completed()
    IC.classifier = joblib.load('trained_model.pkl')
    images = []
    while len(images) < 3:
        if len(images) == 2: robot.turn_in_place(angle = cozmo.util.Angle(degrees = 5), speed = cozmo.util.Angle(degrees = 5)).wait_for_completed()
        else: robot.turn_in_place(angle = cozmo.util.Angle(degrees = 5), speed = cozmo.util.Angle(degrees = 5)).wait_for_completed()
        images.append(numpy.asarray(robot.world.latest_image.raw_image))
    labels = IC.predict_labels(IC, IC.extract_image_features(IC.classifier, images))
    label = max(set(labels), key=list(labels).count)
    FSM(robot, label)

def order(robot: cozmo.robot.Robot):

    # Place a cube at point C on the arena. Start your robot at point D on the arena and directly
    # face the cube. The robot should locate the cube (any cube if you have more than one), pick up the
    # cube, drive forward with the cube to the end of the arena (point A), put down the cube, and drive
    # backward to the robot’s starting location. Then return to the Idle state. 

    robot.say_text(str('order')).wait_for_completed()
    robot.set_lift_height(height=0).wait_for_completed()
    cube = robot.world.wait_for_observed_light_cube(timeout=5)
    robot.pickup_object(cube, num_retries=2).wait_for_completed()
    robot.drive_straight(distance_mm(275), speed_mmps(77)).wait_for_completed()
    robot.place_object_on_ground_here(cube).wait_for_completed()
    robot.drive_straight(distance_mm(-275), speed_mmps(77)).wait_for_completed()


def drone(robot: cozmo.robot.Robot):

    # Have the robot drive in an “S” formation. Show an animation of your choice on the
    # robot’s face. Then return to the Idle state.

    robot.say_text(str('drone')).wait_for_completed()
    trigger =  cozmo.anim.Triggers.CodeLabAmazed
    robot.play_anim_trigger(trigger, in_parallel=True)
    robot.drive_wheels(l_wheel_speed=10, r_wheel_speed=30, duration=10)
    robot.play_anim_trigger(trigger, in_parallel=True)
    robot.drive_wheels(l_wheel_speed=30, r_wheel_speed=10, duration=10)
    robot.play_anim_trigger(trigger, in_parallel=True).wait_for_completed()                         
def inspection(robot: cozmo.robot.Robot):
    
    # Have the robot drive in a square, where each side of the square is approximately 20 cm.
    # While driving, the robot must continuously raise and lower the lift, but do so slowly (2-3 seconds
    # to complete lowering or raising the lift). Simultaneously, the robot must say, “I am not a spy”.
    # Lower the lift at the end of the behavior, and return to the Idle state.

    robot.say_text(str('inspection')).wait_for_completed()
    i = 0
    while i < 4:
        if robot.lift_ratio > .5:
            robot.move_lift(-0.33)
        else:
            robot.move_lift(0.33)

        robot.say_text(str('I am not a spy'))
        robot.drive_straight(distance_mm(200), speed_mmps(77), in_parallel = True).wait_for_completed()
        robot.turn_in_place(degrees(90), in_parallel = True).wait_for_completed()
        i = i + 1

def main(robot: cozmo.robot.Robot):
    run(robot)
    idle(robot)
cozmo.run_program(main)