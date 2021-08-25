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

flow_h5_path = 'flow/SHT_Flows.h5'
flow_h5_keys = list(h5py.File(flow_h5_path, 'r').keys())
# print(rgb_h5_keys)



rgb_h5_path = '../AllDatasets/SHT_Frames.h5'
rgb_h5_keys = list(h5py.File(rgb_h5_path, 'r').keys())

with h5py.File(flow_h5_path, 'r') as h5:
    for rgb_key in rgb_h5_keys:

        if rgb_key not in flow_h5_keys:
            print(rgb_key)
            print('yes')


# key = h5_keys[0]
# print(key)
# with h5py.File(h5_path, 'r') as h5:
#     data = h5[key][:]
#     print(data.shape)
