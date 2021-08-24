import numpy as np
import os
import yaml
import h5py


#
# config = yaml.load(open('configs/zwh_i3d.yaml', 'r'), Loader=yaml.FullLoader)
#
# print(config['class_reweights'][0])
#
# config['gpu'] = [i for i in range(len('1,2,3,4'.split(',')))]
# print(config['gpu'])

h5_path = 'flow/SHT_Flows.h5'
print(h5py.File(h5_path, 'r'))
h5_keys = list(h5py.File(h5_path, 'r').keys())
key = h5_keys[0]
print(key)
with h5py.File(h5_path, 'r') as h5:
    data = h5[key][:]
    print(data.shape)
