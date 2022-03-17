#!/usr/bin/env python

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

_HOST_ = '127.0.0.1'
_PORT_ = 2000
_SLEEP_TIME_ = 1


def main():
	client = carla.Client(_HOST_, _PORT_)
	client.set_timeout(3.0)
	world = client.get_world()
	
	while(True):
		t = world.get_spectator().get_transform()
		coordinate_str = "(x,y,z) = ({},{},{})".format(t.location.x, t.location.y,t.location.z)
		rotation_str = "(pitch,yaw,roll) = ({},{},{})".format(t.rotation.pitch, t.rotation.yaw, t.rotation.roll)
		print (coordinate_str)
		print (rotation_str)
		time.sleep(_SLEEP_TIME_)



if __name__ == '__main__':
	main()