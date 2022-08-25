import cv2
import os
import numpy as np
import paddle


class COCOValDataset(paddle.io.Dataset):
    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 img_size=[640, 640],
                 input_name='x2paddle_images'):
        from pycocotools.coco import COCO
        self.dataset_dir = dataset_dir
        self.image_dir = image_dir
        self.img_size = img_size
        self.input_name = input_name
        self.ann_file = os.path.join(dataset_dir, anno_path)
        self.coco = COCO(self.ann_file)
        ori_ids = list(sorted(self.coco.imgs.keys()))
        # check gt bbox
        clean_ids = []
        for idx in ori_ids:
            ins_anno_ids = self.coco.getAnnIds(imgIds=[idx], iscrowd=False)
            instances = self.coco.loadAnns(ins_anno_ids)
            num_bbox = 0
            for inst in instances:
                if inst.get('ignore', False):
                    continue
                if 'bbox' not in inst.keys():
                    continue
                elif not any(np.array(inst['bbox'])):
                    continue
                else:
                    num_bbox += 1
            if num_bbox > 0:
                clean_ids.append(idx)
        self.ids = clean_ids

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = self._get_img_data_from_img_id(img_id)
        img, scale_factor = self.image_preprocess(img, self.img_size)
        return {
            'image': img,
            'im_id': np.array([img_id]),
            'scale_factor': scale_factor
        }

    def __len__(self):
        return len(self.ids)

    def _get_img_data_from_img_id(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.dataset_dir, self.image_dir,
                                img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _generate_scale(self, im, target_shape, keep_ratio=True):
        """
            Args:
                im (np.ndarray): image (np.ndarray)
            Returns:
                im_scale_x: the resize ratio of X
                im_scale_y: the resize ratio of Y
            """
        origin_shape = im.shape[:2]
        if keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(target_shape)
            target_size_max = np.max(target_shape)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = target_shape
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x

    def image_preprocess(self, img, target_shape):
        # Resize image
        im_scale_y, im_scale_x = self._generate_scale(img, target_shape)
        img = cv2.resize(
            img,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=cv2.INTER_LINEAR)
        # Pad
        im_h, im_w = img.shape[:2]
        h, w = target_shape[:]
        if h != im_h or w != im_w:
            canvas = np.ones((h, w, 3), dtype=np.float32)
            canvas *= np.array([114.0, 114.0, 114.0], dtype=np.float32)
            canvas[0:im_h, 0:im_w, :] = img.astype(np.float32)
            img = canvas
        img = np.transpose(img / 255, [2, 0, 1])
        scale_factor = np.array([im_scale_y, im_scale_x])
        return img.astype(np.float32), scale_factor


class COCOTrainDataset(COCOValDataset):
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = self._get_img_data_from_img_id(img_id)
        img, scale_factor = self.image_preprocess(img, self.img_size)
        return {self.input_name: img}


def _generate_scale(im, target_shape):
    origin_shape = im.shape[:2]
    im_size_min = np.min(origin_shape)
    im_size_max = np.max(origin_shape)
    target_size_min = np.min(target_shape)
    target_size_max = np.max(target_shape)
    im_scale = float(target_size_min) / float(im_size_min)
    if np.round(im_scale * im_size_max) > target_size_max:
        im_scale = float(target_size_max) / float(im_size_max)
    im_scale_x = im_scale
    im_scale_y = im_scale
    return im_scale_y, im_scale_x


def yolo_image_preprocess(img, target_shape=[640, 640]):
    # Resize image
    im_scale_y, im_scale_x = _generate_scale(img, target_shape)
    img = cv2.resize(
        img,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=cv2.INTER_LINEAR)
    # Pad
    im_h, im_w = img.shape[:2]
    h, w = target_shape[:]
    if h != im_h or w != im_w:
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array([114.0, 114.0, 114.0], dtype=np.float32)
        canvas[0:im_h, 0:im_w, :] = img.astype(np.float32)
        img = canvas
    img = np.transpose(img / 255, [2, 0, 1])
    return img.astype(np.float32)
