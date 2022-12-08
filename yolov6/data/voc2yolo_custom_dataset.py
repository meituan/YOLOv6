from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
from yaml.loader import SafeLoader
import xml.etree.ElementTree as ElementTree
from tqdm import tqdm
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

"""
Organize your directory of custom dataset as follows:
    data_custom
    ├───train
    │   ├───images
    │   │   ├───image_xxx.png
    │   │   └───image_xxx.png ...
    │   ├───labels_xml
    │   │   ├───label_xxx.xml
    │   │   └───label_xxx.xml ...
    │   └───labels_txt
    ├───test
    │   ├───images
    │   │   ├───image_xxx.png
    │   │   └───image_xxx.png ...
    │   ├───labels_xml
    │   │   ├───label_xxx.xml
    │   │   └───label_xxx.xml ...
    │   └───labels_txt
    └───val
        ├───images
        │   ├───image_xxx.png
        │   └───image_xxx.png ...
        ├───labels_xml
        │   ├───label_xxx.xml
        │   └───label_xxx.xml ...
        └───labels_txt
"""


def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_label(xml_lb_dir, txt_lb_dir, image_id, voc_names):
    in_file = open(os.path.join(xml_lb_dir, f'{image_id}.xml'))
    out_file = open(os.path.join(txt_lb_dir, f'{image_id}.txt'), 'w')
    tree = ElementTree.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in voc_names and not int(obj.find('difficult').text) == 1:
            xml_box = obj.find('bndbox')
            bb = convert_box((w, h), [float(xml_box.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = voc_names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


def convert_labels_xml2txt(path_data, voc_class):
    dir_labels_txt = os.path.join(path_data, 'labels_txt')
    dir_labels_xml = os.path.join(path_data, 'labels_xml')
    if not os.path.exists(dir_labels_txt):
        os.mkdir(dir_labels_txt)
        print("created folder : ", dir_labels_txt)
    list_labels_xml = os.listdir(dir_labels_xml)
    for i in tqdm(range(0, len(list_labels_xml))):
        file_id = list_labels_xml[i].strip().split(".")[0]
        convert_label(xml_lb_dir=dir_labels_xml, txt_lb_dir=dir_labels_txt, image_id=file_id, voc_names=voc_class)
        pass
    pass


def runs(dict_folder, voc_name_of_class):
    for folder in dict_folder:
        if os.path.exists(dict_folder[folder]):
            print("========== Start processing folder", dict_folder[folder], "============")
            convert_labels_xml2txt(path_data=dict_folder[folder], voc_class=voc_name_of_class)
            print("========== End processing folder", dict_folder[folder], "============")
        else:
            print("No such file or directory: ", dict_folder[folder])
    pass


def main(config):
    args = vars(config.parse_args())
    data_yaml = args["path_yaml"]
    if isinstance(data_yaml, str):
        with open(data_yaml, errors='ignore') as f:
            data_dict = yaml.load(f, Loader=SafeLoader)
            runs(
                dict_folder={
                    'train': data_dict['train'],
                    'val': data_dict['val'],
                    'test': data_dict['test'],
                },
                voc_name_of_class=data_dict['names']
            )
    pass


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path_yaml', type=str, default='./data/dataset.yaml', help='from root to path data.yaml')
    print("==========start convert labels VOC xml to txt=============")
    main(parser)
    print("==========End convert labels VOC xml to txt=============")
    pass
