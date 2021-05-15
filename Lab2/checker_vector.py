import asyncio
import concurrent
import anki_vector
from anki_vector.events import Events
from anki_vector import annotate
from anki_vector.util import degrees, distance_mm, speed_mmps
import time
from itertools import chain
import sys
import datetime

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Cannot import from PIL: Do `pip3 install --user Pillow` to install")

gyro = []
acc = []
class RobotStateDisplay(annotate.Annotator):
    def apply(self, image, scale):
        d = ImageDraw.Draw(image)

        bounds = [3, 0, image.width, image.height]

        def print_line(text_line):
            text = annotate.ImageText(text_line, position=annotate.AnnotationPosition.TOP_LEFT, outline_color='black', color='lightblue')
            text.render(d, bounds)
            TEXT_HEIGHT = 11
            bounds[1] += TEXT_HEIGHT

        robot = self.world.robot

        pose = robot.pose
        print_line('Pose: Pos = <%.1f, %.1f, %.1f>' % pose.position.x_y_z)
        print_line('Pose: Rot quat = <%.1f, %.1f, %.1f, %.1f>' % pose.rotation.q0_q1_q2_q3)
        print_line('Pose: angle_z = %.1f' % pose.rotation.angle_z.degrees)
        print_line('Pose: origin_id: %s' % pose.origin_id)

        # Display the Accelerometer and Gyro data for the robot
        print_line('Accelmtr: <%.1f, %.1f, %.1f>' % robot.accel.x_y_z)
        print_line('Gyro: <%.1f, %.1f, %.1f>' % robot.gyro.x_y_z)
        gyro.append(robot.gyro.x_y_z)
        acc.append(robot.accel.x_y_z)


async def main():
    #args = anki_vector.util.parse_command_args()
    arg = sys.argv[1]
    print(arg)
    with anki_vector.Robot(serial=arg,show_viewer=True) as robot:
        print("Vector SDK has behavior control...")
        robot.behavior.set_head_angle(degrees(0))
        robot.behavior.say_text("Hello World!")
        time.sleep(1)
        robot.behavior.say_text("Let's take pictures!")
        time.sleep(1)
        
        # take pictures
        myargs = sys.argv[2:]
        if len(myargs) <= 1:
            sys.exit("Incorrect arguments")
        
        num_images_per_type = int(myargs[0])  # number of images to take of each type of object
        
        print("Taking ", num_images_per_type, "images each of ", myargs[1:])

        for type in myargs[1:]:
            for i in range(num_images_per_type):
                time.sleep(.5)
                robot.behavior.say_text(type)
                time.sleep(3)
                latest_image = robot.camera.latest_image
                new_image = latest_image.raw_image
                timestamp = datetime.datetime.now().strftime("%dT%H%M%S%f")
                new_image.save("./outputs/" + str(type) + "_" + timestamp + ".bmp")
                robot.behavior.say_text("Picture taken")
                time.sleep(3)
        robot.camera.image_annotator.add_annotator('robotState', RobotStateDisplay)
    # physical tests
        robot.behavior.say_text("I will drive now")
        time.sleep(1.5)
        robot.behavior.drive_straight(distance_mm(1000), speed_mmps(1000))
        robot.behavior.say_text("Check out my arms")
        robot.behavior.set_lift_height(0.0)
        robot.behavior.set_lift_height(1.0)
        robot.behavior.set_lift_height(0.0)
        robot.behavior.say_text("Pick me up and shake me around")
        time.sleep(5)
        robot.behavior.say_text("You can put me down now")
        time.sleep(0.5)
        f = open("./outputs/test_output.txt", 'w')
        f.seek(0)
        f.write("Min Gyro: " + str(min(gyro)) + "\n")
        f.write("Max Gyro: " + str(max(gyro)) + "\n")
        f.write("Min Acc: " + str(min(acc)) + "\n")
        f.write("Max Acc: " + str(max(acc)))
        f.truncate()
        f.close()
        if (all(i > -10 for i in min(gyro)) and all(i < 10 for i in max(gyro))) and (all(i > -25000 for i in min(acc)) and all(i < 25000 for i in max(acc))):
            robot.behavior.say_text("Sensors look good!")
        else:
            robot.behavior.say_text("Sensors look bad!")

if __name__ == "__main__":
    asyncio.run(main())
