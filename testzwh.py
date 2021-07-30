import numpy as np
import os
import yaml


config = yaml.load(open('configs/zwh_i3d.yaml', 'r'), Loader=yaml.FullLoader)

print(config['class_reweights'][0])
