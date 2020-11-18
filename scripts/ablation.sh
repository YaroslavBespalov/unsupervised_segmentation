#!/usr/bin/env bash

python3.6 /home/ibespalov/unsupervised_pattern_segmentation/examples/change_params.py  -faked=0.0 -Rb=14.0 -Rt=1500.0 -L1image=0.0 -fakecontloss=0.0
python3.6 /home/ibespalov/unsupervised_pattern_segmentation/examples/stylegan_train_unsupervised.py

python3.6 /home/ibespalov/unsupervised_pattern_segmentation/examples/change_params.py  -faked=0.14 -Rb=14.0 -Rt=1500.0 -L1image=0.0 -fakecontloss=0.0
python3.6 /home/ibespalov/unsupervised_pattern_segmentation/examples/stylegan_train_unsupervised.py

python3.6 /home/ibespalov/unsupervised_pattern_segmentation/examples/change_params.py  -faked=0.14 -Rb=14.0 -Rt=1500.0 -L1image=70.0 -fakecontloss=0.0
python3.6 /home/ibespalov/unsupervised_pattern_segmentation/examples/stylegan_train_unsupervised.py

python3.6 /home/ibespalov/unsupervised_pattern_segmentation/examples/change_params.py  -faked=0.14 -Rb=14.0 -Rt=1500.0 -L1image=70.0 -fakecontloss=50.0
python3.6 /home/ibespalov/unsupervised_pattern_segmentation/examples/stylegan_train_unsupervised.py



