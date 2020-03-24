## Generic code for data preparation
## Requires the data to be partitioned into gallery and probe sets, subject wise (in folders), with same name
## This only contains gallery and probe data only, test data has to be kept separate


import os
import numpy as np
from scipy import misc

r = []
rLabel = []
t = []
tLabel = []
count = 1
lab = 1
dsDir = '/home/vplab/Avishek/DATA/' ## Directory to data
trdir_ = os.path.join(dsDir, 'prb/') # Directory to probe data
tsdir_ = os.path.join(dsDir, 'gal/') # Directory to gallery data

for shots in sorted(os.listdir(trdir_)): # Navigate folder wise in both gallery and probe folders
    trdir = os.path.join(trdir_, shots) # Folder names should be same in both gallery and probe folders
    tsdir = os.path.join(tsdir_, shots)
    num_files_ts = len([name for name in os.listdir(tsdir)])
    num_files_tr = len([name for name in os.listdir(trdir)])
    tslist = sorted(os.listdir(tsdir))
    trlist = sorted(os.listdir(trdir))
    print("\n")
    count = 0
    overall_count = 0
    print("Starting %d" % int(shots))
    for files in sorted(os.listdir(trdir)):
        shotDir_tr = trdir
        filepath_tr = os.path.join(shotDir_tr, files)
        filepath_ts = os.path.join(tsdir, tslist[count % num_files_ts])
        if overall_count % 20 == 0:
            if os.path.exists(filepath_tr):
                img_tr = misc.imread(filepath_tr)
                img_tr = misc.imresize(img_tr, [35, 35, 3], 'bicubic')
                img_tr = (img_tr.astype(np.float32) - 127.5) / 127.5

                img_ts = misc.imread(filepath_ts)
                img_ts = misc.imresize(img_ts, [140, 140, 3], 'bicubic')
                img_ts = (img_ts.astype(np.float32) - 127.5) / 127.5

                r.append(img_tr)
                t.append(img_ts)
                temp = np.zeros([30], dtype='float32')
                temp[int(shots)-1] = 1.
                rLabel.append(temp)
                tLabel.append(temp)
            count = count + 1
        overall_count = overall_count + 1
    print("Ending %d" % int(shots))


trImages = np.asarray(r).astype(np.float32)
trLabel = np.asarray(rLabel).astype(np.float32)
tsImages = np.asarray(t).astype(np.float32)
tsLabel = np.asarray(tLabel).astype(np.float32)
print(np.shape(trImages), np.shape(trLabel), np.shape(tsImages), np.shape(tsLabel))
np.save('tsImages_cad_2', tsImages)
np.save('tsLabels_cad_2', tsLabel)
np.save('trImages_cad_2', trImages)
np.save('trLabels_cad_2', trLabel)
