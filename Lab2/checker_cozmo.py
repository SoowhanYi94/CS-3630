
import asyncio
import concurrent
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
import time
from itertools import chain
import sys
import datetime
import time

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Cannot import from PIL: Do `pip3 install --user Pillow` to install")

gyro = []
acc = []

class RobotStateDisplay(cozmo.annotate.Annotator):
    def apply(self, image, scale):
        d = ImageDraw.Draw(image)

        bounds = [3, 0, image.width, image.height]

        def print_line(text_line):
            text = cozmo.annotate.ImageText(text_line, position=cozmo.annotate.TOP_LEFT, outline_color='black', color='lightblue')
            text.render(d, bounds)
            TEXT_HEIGHT = 11
            bounds[1] += TEXT_HEIGHT

        robot = self.world.robot  # type: cozmo.robot.Robot

        # Display the Pose info for the robot

        pose = robot.pose
        print_line('Pose: Pos = <%.1f, %.1f, %.1f>' % pose.position.x_y_z)
        print_line('Pose: Rot quat = <%.1f, %.1f, %.1f, %.1f>' % pose.rotation.q0_q1_q2_q3)
        print_line('Pose: angle_z = %.1f' % pose.rotation.angle_z.degrees)
        print_line('Pose: origin_id: %s' % pose.origin_id)
        gyro.append(robot.gyro.x_y_z)
        acc.append(robot.accelerometer.x_y_z)
        # Display the Accelerometer and Gyro data for the robot

        print_line('Accelmtr: <%.1f, %.1f, %.1f>' % robot.accelerometer.x_y_z)
        print_line('Gyro: <%.1f, %.1f, %.1f>' % robot.gyro.x_y_z)

        # Display the Accelerometer and Gyro data for the mobile device

        if robot.device_accel_raw is not None:
            print_line('Device Acc Raw: <%.2f, %.2f, %.2f>' % robot.device_accel_raw.x_y_z)
        if robot.device_accel_user is not None:
            print_line('Device Acc User: <%.2f, %.2f, %.2f>' % robot.device_accel_user.x_y_z)
        if robot.device_gyro is not None:
            mat = robot.device_gyro.to_matrix()
            print_line('Device Gyro Up: <%.2f, %.2f, %.2f>' % mat.up_xyz)
            print_line('Device Gyro Fwd: <%.2f, %.2f, %.2f>' % mat.forward_xyz)
            print_line('Device Gyro Left: <%.2f, %.2f, %.2f>' % mat.left_xyz)

def cozmo_program(robot: cozmo.robot.Robot):
        print("Cozmo SDK has behavior control...")
        robot.world.image_annotator.add_annotator('robotState', RobotStateDisplay)
        robot.enable_device_imu(True, True, True)
        robot.camera.image_stream_enabled = True
        robot.camera.color_image_enabled = False
        robot.camera.enable_auto_exposure()
        robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
        robot.move_lift(-1)

        robot.say_text("Hello World!").wait_for_completed()
        time.sleep(1)
        robot.say_text("Let's take pictures!").wait_for_completed()
        time.sleep(1)

        # take pictures
        # myargs = ["1", "order", "drone", "inspection"]
        myargs = sys.argv[1:]
        
        if len(myargs) <= 1:
            sys.exit("Incorrect arguments")
        
        num_images_per_type = int(myargs[0])  # number of images to take of each type of object
        
        print("Taking ", num_images_per_type, "images each of ", myargs[1:])

        for type in myargs[1:]:
            for i in range(num_images_per_type):
                time.sleep(.5)

                robot.say_text(type).wait_for_completed()

                time.sleep(3)

                latest_image = robot.world.latest_image
                new_image = latest_image.raw_image

                timestamp = datetime.datetime.now().strftime("%dT%H%M%S%f")

                new_image.save("./outputs/" + str(type) + "_" + timestamp + ".bmp")

                time.sleep(3)
        time.sleep(0.5)
        robot.say_text("I will drive now!").wait_for_completed()
        time.sleep(0.5)
        robot.drive_straight(distance_mm(1000), speed_mmps(1000)).wait_for_completed()
        robot.say_text("Check out my arms!").wait_for_completed()
        robot.move_lift(1)
        time.sleep(1)
        robot.move_lift(-1)
        time.sleep(1)
        robot.say_text("Pick me up and shake me around!").wait_for_completed()
        time.sleep(2)
        robot.say_text("You can put me down now!").wait_for_completed()
        f = open("./outputs/test_output.txt", 'r+')
        f.seek(0)
        f.write("Min Gyro: " + str(min(gyro)) + "\n")
        f.write("Max Gyro: " + str(max(gyro)) + "\n")
        f.write("Min Acc: " + str(min(acc)) + "\n")
        f.write("Max Acc: " + str(max(acc)))
        f.truncate()
        f.close()
        if (all(i > -10 for i in min(gyro)) and all(i < 10 for i in max(gyro))) and (all(i > -25000 for i in min(acc)) and all(i < 25000 for i in max(acc))):
            robot.say_text("Sensors look good!").wait_for_completed()
        else:
            robot.say_text("Sensors look bad!").wait_for_completed()

cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)