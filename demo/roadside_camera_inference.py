import glob
import os
import sys
from collections import deque
try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    # sys.path.append('D:\CODE\Python\CARLA\CarlaSimulator\PythonAPI\examples\custom\carla-0.9.10-py3.7-win-amd64.egg')
except IndexError:
    print("Carla not found")
    pass
import carla

import time
import darknet
import carlaUtil as cu

IM_WIDTH = 1024
IM_HEIGHT = 682

# If you want to use a fixed camera location, uncomment the following lines

# CAM_LOC = (-62.29660415649414,-7.814524173736572,3.642631769180298)
# CAM_ROT = (-6.816617012023926,144.29969787597656,3.0309773137560114e-05)

# Change these paths

CFG_PATH = "D:\CODE\Python\CARLA\yolov4-tiny\darknet-build\Release\cfg\yolov4-tiny-custom.cfg"
DATA_PATH = "D:\CODE\Python\CARLA\yolov4-tiny\darknet-build\Release\data\obj.data"
WEIGHTS_PATH = "D:\CODE\Python\CARLA\yolov4-tiny\weights\yolov4-tiny-kaggle-and-angled.weights"


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("sensor_tick", "5.0")
    
    # Camera uses spectator's location and rotation
    t = world.get_spectator().get_transform()

    # If you want to use a fixed camera location, uncomment the following lines
    # sp = carla.Location(CAM_LOC[0], CAM_LOC[1], CAM_LOC[2])
    # cam_rotation = carla.Rotation(CAM_ROT[0], CAM_ROT[1], CAM_ROT[2])

    sp = carla.Location(t.location.x, t.location.y, t.location.z)
    cam_rotation = carla.Rotation(t.rotation.pitch, t.rotation.yaw, t.rotation.roll)
    cam_transform = carla.Transform(sp,cam_rotation)

    ego_cam = world.spawn_actor(cam_bp,cam_transform)

    actor_list.append(ego_cam)

    # Must load network before using dark_inference
    net, cn, color = darknet.load_network(CFG_PATH, DATA_PATH, WEIGHTS_PATH)

    q = deque(maxlen=5000)
    ego_cam.listen(lambda data: q.append(data))
    while 1:
        if q:
            cu.dark_inference(q.popleft(), net, cn, color)

    time.sleep(300)
    

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')