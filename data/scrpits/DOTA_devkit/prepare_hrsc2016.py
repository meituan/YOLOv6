import os
import os.path as osp
from DOTA_devkit.HRSC2DOTA import generate_txt_labels
from DOTA_devkit.DOTA2JSON import generate_json_labels

def preprare_hrsc2016(data_dir):
    train_dir = osp.join(data_dir,'Train')
    test_dir = osp.join(data_dir, 'Test')
    # convert hrsc2016 to dota raw format
    generate_txt_labels(train_dir)
    generate_txt_labels(test_dir)
    # convert it to json format
    generate_json_labels(train_dir,osp.join(train_dir,'trainval.json'))
    generate_json_labels(test_dir,osp.join(test_dir,'test.json'), trainval=False)

if __name__ == '__main__':
    hrsc2016_dir = '/mnt/SSD/lwt_workdir/BeyondBoundingBox/data/hrsc2016/'
    preprare_hrsc2016(hrsc2016_dir)
    print('done')
