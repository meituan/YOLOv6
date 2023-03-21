import dota_utils as util
import os
import cv2
import json
from PIL import Image
import os.path as osp

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
               
wordname_18 = [ 'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
         'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor',
         'swimming-pool', 'helicopter', 'container-crane', 'airport', 'helipad']
               
hrsc_2016 = ['ship']
ucas_aod = ['car', 'airplane']

def DOTA2COCOTrain(srcpath, destfile, cls_names, difficult='2'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    print('difficult: ', difficult)
                    continue
                single_obj = {}
                single_obj['area'] = obj['area']

                if obj['name'] not in cls_names: # 去掉别的类
                    print('This classname not found: ', obj['name'])
                    continue
                
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                # xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                #                          max(obj['poly'][0::2]), max(obj['poly'][1::2])
                x1 = obj['poly'][0]
                y1 = obj['poly'][1]
                x2 = obj['poly'][2]
                y2 = obj['poly'][3]
                x3 = obj['poly'][4]
                y3 = obj['poly'][5]
                x4 = obj['poly'][6]
                y4 = obj['poly'][7]

                # width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = x1, y1, x2, y2, x3, y3, x4, y4
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)

def DOTA2COCOTest(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(imageparent)
        for file in filenames:
            basename = util.custombasename(file)
            imagepath = os.path.join(imageparent, basename + '.png')
            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)

if __name__ == '__main__':

    DOTA2COCOTrain(r'data/dataset_demo_split',
                  osp.join('data/dataset_demo_split', 'train_datasetdemo.json'),
                  wordname_15)
    print('Train DONE')
    # DOTA2COCOTrain(r'/dataset/Dota/Dota_V1.0/train_sp/',
    #              r'/mnt/SSD/lwt_workdir/s2anet_branch/data/ucas_aod/Train/trainval_coco_8point.json',
    #              ucas_aod)
    DOTA2COCOTest(r'data/dataset_demo_split',
                 osp.join('data/dataset_demo_split', 'test_datasetdemo.json'),
                 wordname_15)
    # DOTA2COCOTest(r'/mnt/SSD/lwt_workdir/s2anet_branch/data/ucas_aod/Test/',
    #               r'/mnt/SSD/lwt_workdir/s2anet_branch/data/ucas_aod/Test/test_coco_8point.json',
    #               ucas_aod)

    print('Test DONE')
