Code for D2SC-GAN accepted in IEEE T-BIOM.
To run the code, please install the following:

tensorflow-gpu==1.8.0
</br>
keras==2.1.6
</br>
matplotlib==2.2.4
</br>
Pillow==6.0.0
</br>
pydot==1.4.1
</br>
scipy==1.2.1
</br>
</br>

Also, make sure to install python-tk & graphviz: sudo apt-get install python-tk graphviz

</br>

To prepare the data for training, separate the probe samples into train (15% samples per subject) and test set.
The gallery and probe data folders should have all subject samples, with same names for subject folders. 

For example, if subject 1 is named "001" in gallery folder, it should be named the same in probe (train) folder too.

After data preparation, make changes to point to the correct gallery and probe npy files in "d2scgan_train.py" to run the code. Follow the comments in the code.

To test the trained model, point the path to the test data npy file and run test.py.

The code requires a machine with Nvidia Tesla (32GB) and >=128GB RAM to train and test.

If you are using this code for your work, please cite:
</br>
</br>
@article{bhattacharjee2020d2sc,
</br>
title={D2SC-GAN: Dual Deep-Shallow Channeled Generative Adversarial Network, for Resolving Low-resolution Faces for Recognition in Classroom scenarios},
</br>
author={Bhattacharjee, Avishek and Das, Sukhendu},
</br>
booktitle = { IEEE Transactions on Biometrics, Behavior, and Identity Science},
</br>
month = {March},
</br>
year = {2020}
</br>
}
