import os
import numpy as np
import cv2
import copy
import dota_utils as util
from multiprocessing import Pool
from functools import partial


def split_single_warp(name, split_base, rate, extent):
    split_base.SplitSingle(name, rate, extent)


class splitbase():
    def __init__(self,
                 srcpath,
                 dstpath,
                 gap=100,
                 subsize=1024,
                 ext='.png',
                 padding=True,
                 num_process=32):
        self.srcpath = srcpath
        self.outpath = dstpath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.srcpath = srcpath
        self.dstpath = dstpath
        self.ext = ext
        self.padding = padding
        self.pool = Pool(num_process)

        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)

    def saveimagepatches(self, img, subimgname, left, up, ext='.png'):
        subimg = copy.deepcopy(
            img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.dstpath, subimgname + ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def SplitSingle(self, name, rate, extent):
        img = cv2.imread(os.path.join(self.srcpath, name + extent))
        assert np.shape(img) != ()

        if (rate != 1):
            resizeimg = cv2.resize(
                img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'

        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        # if (max(weight, height) < self.subsize/2):
        #     return

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                subimgname = outbasename + str(left) + '___' + str(up)
                self.saveimagepatches(resizeimg, subimgname, left, up)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):

        imagelist = util.GetFileFromThisRootDir(self.srcpath)
        imagenames = [util.custombasename(x) for x in imagelist if (
            util.custombasename(x) != 'Thumbs')]

        # worker = partial(self.SplitSingle, rate=rate, extent=self.ext)
        worker = partial(split_single_warp, split_base=self,
                         rate=rate, extent=self.ext)
        self.pool.map(worker, imagenames)
        #
        # for name in imagenames:
        #     self.SplitSingle(name, rate, self.ext)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


if __name__ == '__main__':
    split = splitbase(r'/media/test/4d846cae-2315-4928-8d1b-ca6d3a61a3c6/DOTA/DOTAv1.5/test/images', 
                      r'/media/test/4d846cae-2315-4928-8d1b-ca6d3a61a3c6/DOTA/DOTAv2.0/test-dev_split/images',
                      gap=200, subsize=1024, num_process=8)
    split.splitdata(1)
    # split.splitdata(0.5)
    # split.splitdata(1.5)
    print("Split Done!")

