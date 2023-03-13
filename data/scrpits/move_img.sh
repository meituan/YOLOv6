#! /usr/bin/bash
# 转换成yolo格式
# 创建文件夹
mkdir -p images/train
mkdir -p images/val
mkdir -p images/test
mkdir -p labels/train
mkdir -p labels/val
mkdir -p labels/test
mkdir -p Annotations_txt

# 根据原始info移动img
# train 一般与 val 合并, 同放到train中
# 在ImageSets中
# trainval.txt
# 定义旧地址和新地址
old_path="./Images"
new_path_train="./images/train"
new_path_val="./images/val"
new_path_test="./images/test"

#! HRSC2016含有一部分无目标的图片,可能会cp会报错
# 读取image_list.txt，并循环处理每个图片文件
while read -r filename; do
	filename="$filename.png"
	# 复制图片到新地址
	cp "$old_path/$filename" "$new_path_train/$filename"
done <"./ImageSets/trainval.txt"

while read -r filename; do
	filename="$filename.png"
	# 复制图片到新地址
	cp "$old_path/$filename" "$new_path_val/$filename"
done <"./ImageSets/val.txt"

while read -r filename; do
	filename="$filename.png"
	# 复制图片到新地址
	cp "$old_path/$filename" "$new_path_test/$filename"
done <"./ImageSets/test.txt"

# 执行python脚本, 从xml格式转化为yolo格式, 归一化的数据集
python ./HRSC_voc_xml_2_txt_new5_augments.py
# 移动labels

old_path="./Annotations_txt"
new_path_train="./labels/train"
new_path_val="./labels/val"
new_path_test="./labels/test"

#! HRSC2016含有一部分无目标的图片,可能会cp会报错
# 读取image_list.txt，并循环处理每个图片文件
while read -r filename; do
	filename="$filename.txt"
	# 复制图片到新地址
	cp "$old_path/$filename" "$new_path_train/$filename"
done <"./ImageSets/trainval.txt"

while read -r filename; do
	filename="$filename.txt"
	# 复制图片到新地址
	cp "$old_path/$filename" "$new_path_val/$filename"
done <"./ImageSets/val.txt"

while read -r filename; do
	filename="$filename.txt"
	# 复制图片到新地址
	cp "$old_path/$filename" "$new_path_test/$filename"
done <"./ImageSets/test.txt"
