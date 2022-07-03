from setuptools import setup
import os

# collect requirements from requirements.txt
lib_dir = os.path.dirname(os.path.realpath(__file__))
requirements_path = lib_dir + "/requirements.txt"
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

# remove comments
_install_requires = []
for req in install_requires:
    idx = req.find("#")
    req = req[:idx]
    if req:
        _install_requires.append(req)
install_requires = _install_requires

setup(
    name='yolov6',
    version='0.1.0',
    packages=[''],
    url='https://github.com/meituan/YOLOv6',
    license='GNU General Public License v3.0',
    author='meituan',
    author_email='',
    description='',
    install_requires=install_requires
)