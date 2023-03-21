# DOTA_devkit Install

```shell
cd data/scripts/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
