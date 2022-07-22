# %%
import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

import string


import pathlib

random.seed(108)

# %%
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    a=xml_file.split("/")[-1].replace("xml","jpg")
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []
    info_dict['filename']=a
    if root.attrib == {'verified': 'yes'}:

        # Parse the XML Tree
        for elem in root:
            # # Get the file name 
            # if elem.tag == "filename":
            #     info_dict['filename'] = elem.text  
            # Get the image size
            if elem.tag == "size":
                image_size = []
                for subelem in elem:
                    image_size.append(int(subelem.text))
                
                info_dict['image_size'] = tuple(image_size)
            
            # Get details of the bounding box 
            elif elem.tag == "object":
                bbox = {}
                for subelem in elem:
                    if subelem.tag == "name":
                        bbox["class"] = subelem.text
                        
                    elif subelem.tag == "bndbox":
                        for subsubelem in subelem:
                            bbox[subsubelem.tag] = int(subsubelem.text)            
                info_dict['bboxes'].append(bbox)
        
    return info_dict





class_name_to_id_mapping=dict(zip(list("1234567890"+string.ascii_uppercase),list(range(len(list("1234567890"+string.ascii_uppercase))))))


# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
    # print(print_buffer)
    # Name of the file which we have to save 
    save_file_name = os.path.join("unval_add_anno_txt", info_dict["filename"].replace("jpg", "txt"))

    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))






for a,b,c in os.walk("/workspaces/internship_YOLO/yolov5/unval_add_anno/"):
    annotations=[a+i for i in c]

# Convert and save the annotations
for ann in tqdm(annotations):
    info_dict = extract_info_from_xml(ann)
    # print(info_dict)
    convert_to_yolov5(info_dict)


annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]



random.seed(0)

class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()

# Get any random annotation file 
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x ] for x in annotation_list]

#Get the corresponding image file
image_file = annotation_file.replace("annotations", "images").replace("txt", "jpg")
print(image_file)
assert os.path.exists(image_file)

#Load the image
image = Image.open(image_file)

#Plot the Bounding Box
plot_bounding_box(image, annotation_list)

# Read images and annotations
# images = [os.path.join('images', x) for x in os.listdir('images')]
annotations = [os.path.join('annotations', x) for x in os.listdir('annotations') if x[-3:] == "txt"]
images=[a.replace("annotations","images").replace("txt","jpg") for a in annotations]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

len(train_annotations)

# !mkdir images/train images/val images/test annotations/train annotations/val annotations/test


#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'annotations/train/')
move_files_to_folder(val_annotations, 'annotations/val/')
move_files_to_folder(test_annotations, 'annotations/test/')



os.listdir("/workspaces/internship_YOLO/yolov5/unval_add_anno_txt")



