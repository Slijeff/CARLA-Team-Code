from dis import dis
import glob
import os
import sys
from typing import Counter
from collections import deque
try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Carla not found")
    pass
import carla

import random
import time
import numpy as np
import cv2
import torch
import darknet
import matplotlib.pyplot as plt

IM_WIDTH = 1024
IM_HEIGHT = 1024

CAM_LOC = (-63.32263946533203,0.13137656450271606,5.5841593742370605)
CAM_ROT = (-10.645600318908691,-156.46551513671875,-0.0001828667300287634)

CFG_PATH = "D:\CODE\Python\CARLA\yolov4-tiny\darknet-build\Release\cfg\yolov4-tiny-custom.cfg"
DATA_PATH = "D:\CODE\Python\CARLA\yolov4-tiny\darknet-build\Release\data\obj.data"
WEIGHTS_PATH = "D:\CODE\Python\CARLA\CarlaSimulator\PythonAPI\examples\custom\yolov4-tiny-custom_best.weights"

def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def dark_img(data, network, cn, color):
    """
    Inference on trained model and display it as live video.
    """
    rgb_arr = to_bgra_array(data)
    rgb_arr = np.ascontiguousarray(rgb_arr, dtype=np.uint8)
    img, arr = darknet.array_to_image(rgb_arr)
    fps_time = time.time()
    disp = darknet.draw_boxes(darknet.detect_image(network, cn, img), rgb_arr, color)
    cv2.putText(disp, "FPS: {:.1f}".format(1 / (time.time() - fps_time)), (10, 50), 0, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("darknet", disp)
    cv2.waitKey(25)

actor_list = []
try:
    client = carla.Client('localhost', 2000) # https://carla.readthedocs.io/en/0.9.11/core_world/#the-client
    client.set_timeout(3.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library() # https://carla.readthedocs.io/en/0.9.11/core_actors/#blueprints

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("sensor_tick", "5.0")
    
    sp = carla.Location(CAM_LOC[0], CAM_LOC[1], CAM_LOC[2])
    cam_rotation = carla.Rotation(CAM_ROT[0], CAM_ROT[1], CAM_ROT[2])
    cam_transform = carla.Transform(sp,cam_rotation)

    ego_cam = world.spawn_actor(cam_bp,cam_transform)

    actor_list.append(ego_cam)

    net, cn, color = darknet.load_network(CFG_PATH, DATA_PATH, WEIGHTS_PATH)

    q = deque(maxlen=5000)
    ego_cam.listen(lambda data: q.append(data))
    while 1:
        if q:
            t0 = time.time()
            dark_img(q.popleft(), net, cn, color)

    time.sleep(300)
    

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')