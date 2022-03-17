import os

# Path to current labels for that position
label_path = "D:\CODE\Python\CARLA\Data\Labeled\Position 1"
# Path to all images for that position
image_path = "D:\CODE\Python\CARLA\Data\Originals\Position1"
# Where to generate empty labels
generate_path = "D:\CODE\Python\CARLA\Data\Labeled\Position 1"

has_label = set({})
labels = os.listdir(label_path)
for label in labels:
  label = label.split(".")[0]
  has_label.add(label)

images = os.listdir(image_path)
for image in images:
  image = image.split(".")[0]
  if image not in has_label:
    full_path = generate_path + '\\' + image + ".txt"
    fp = open(full_path, 'x')
    fp.close()
    print(f'{image}.txt is created in {full_path}')