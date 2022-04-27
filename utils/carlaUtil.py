import sys
try:
    sys.path.append(
        'D:\CODE\Python\CARLA\CarlaSimulator\PythonAPI\examples\custom\carla-0.9.10-py3.7-win-amd64.egg')
except IndexError:
    print("Carla not found")
    pass
import carla
import numpy as np
import darknet
import time
import cv2


def calculate_distance(ego_loc, car_loc):
    """
    Calculate the distance between two actors in CARLA coordinates.
    """

    coord1 = (ego_loc.x, ego_loc.y, ego_loc.z)
    coord2 = (car_loc.x, car_loc.y, car_loc.z)
    return np.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 + (coord1[2]-coord2[2])**2)


def _to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array

def dark_inference(data, network, cn, color):
    """
    Inference on trained model and display it as live video.
    """
    rgb_arr = _to_bgra_array(data)
    rgb_arr = np.ascontiguousarray(rgb_arr, dtype=np.uint8)
    img, arr = darknet.array_to_image(rgb_arr)
    fps_time = time.time()
    list1 = darknet.detect_image(network, cn, img)
    disp = darknet.draw_boxes(list1, rgb_arr, color)
    cv2.putText(disp, "FPS: {:.1f}".format(1 / (time.time() - fps_time)), (10, 50), 0, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("darknet", disp)
    cv2.waitKey(25)