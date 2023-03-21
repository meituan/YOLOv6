import os
import argparse
import shutil

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

def results_ensemble(srcpath_1, srcpath_2, dstpath):
    """
    将srcpath_1,srcpath_2文件夹中的所有txt中的目标提取出来, 并叠加在一起存入 dstpath
    """
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folderX
    os.makedirs(dstpath)

    filelist_1 = GetFileFromThisRootDir(srcpath_1)  # srcpath文件夹下的所有文件相对路径 eg:['Task1_??.txt', ..., '?.txt']
    filelist_2 = GetFileFromThisRootDir(srcpath_2) 
    for index, fullname_1 in enumerate(filelist_1):  # Task1_??.txt'
        fullname_2 = filelist_2[index]
        basename = custombasename(fullname_1)  # 只留下文件名 eg:'Task1_??'
        dstname = os.path.join(dstpath, basename + '.txt')  # eg: ./.../Task1_plane.txt

        with open(dstname, 'a') as f_out:
            # merge first txt
            with open(fullname_1, 'r') as f1:
                lines = f1.readlines()
                for line in lines:
                    f_out.writelines(line)
            # merge second txt
            with open(fullname_2, 'r') as f2:
                lines = f2.readlines()
                for line in lines:
                    f_out.writelines(line)
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='model ensemble')
    parser.add_argument('--srcpath_1', default='/OrientedRepPoints/tools/parse_pkl/evaluation_results/ORR_results/', help='srcpath_1')
    parser.add_argument('--srcpath_2', default='/OrientedRepPoints/tools/parse_pkl/evaluation_results/ROI_results/', help='srcpath_2')
    parser.add_argument('--dstpath', default='/OrientedRepPoints/tools/parse_pkl/evaluation_results/orientedreppoints_ROIRT_ensemble/', help='dstpath')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    srcpath_1 = args.srcpath_1
    srcpath_2 = args.srcpath_2
    dstpath = args.dstpath

    results_ensemble(srcpath_1, srcpath_2, dstpath)


if __name__ == '__main__':
    main()