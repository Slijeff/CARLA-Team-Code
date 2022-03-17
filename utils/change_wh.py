import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
# change all width and height attributes in the document to 640x480
PATH = './'
files = [f for f in listdir(PATH) if isfile(join(PATH, f))]

for file in files:
    if file.endswith('.xml'):
        tree = ET.parse(file)
        root = tree.getroot()
        root.find('size').find('width').text = '640'
        root.find('size').find('height').text = '480'
        tree.write(file)
        print('Changed: ' + file)