import os
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--srcpath', default=r'/OrientedRepPoints/tools/parse_pkl/evaluation_results/OBB_results/')
    parser.add_argument('--dstpath', default='/OrientedRepPoints/tools/parse_pkl/evaluation_results/HBB_results/',
                        help='dota version')
    args = parser.parse_args()

    return args


def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def OBB2HBB(srcpath, dstpath):
    filenames = GetFileFromThisRootDir(srcpath)
    if os.path.exists(dstpath):
      shutil.rmtree(dstpath)  # delete output folderX
    os.makedirs(dstpath)

    for file in filenames:   # eg: /.../task1_plane.txt

        basename = custombasename(file)  # 只留下文件名 eg:'task1_plane'
        class_basename = basename.split('_')[-1]
        with open(file, 'r') as f_in:
            with open(os.path.join(dstpath, 'Task2_' + class_basename + '.txt'), 'w') as f_out:
                lines = f_in.readlines() 
                splitlines = [x.strip().split() for x in lines]  # list: n*[]
                for index, splitline in enumerate(splitlines):
                    imgname = splitline[0]
                    score = splitline[1]
                    poly = splitline[2:]
                    poly = list(map(float, poly))
                    xmin, xmax, ymin, ymax = min(poly[0::2]), max(poly[0::2]), min(poly[1::2]), max(poly[1::2])
                    rec_poly = [xmin, ymin, xmax, ymax]
                    outline = imgname + ' ' + score + ' ' + ' '.join(map(str, rec_poly))
                    if index != (len(splitlines) - 1):
                        outline = outline + '\n'
                    f_out.write(outline)

if __name__ == '__main__':
    args = parse_args()
    OBB2HBB(args.srcpath, args.dstpath)
