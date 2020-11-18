import torch
from torch import nn, Tensor

import argparse
import json

parser = argparse.ArgumentParser(
                description='Text file conversion.'
                )

parser.add_argument("-path", type=str, default="/home/ibespalov/unsupervised_pattern_segmentation/parameters/content_loss.json")
parser.add_argument("-faked", type=float)
parser.add_argument("-reald", type=float)
parser.add_argument("-sparse", type=float)
parser.add_argument("-Rb", type=float)
parser.add_argument("-Rt", type=float)
parser.add_argument("-L1image", type=float)
parser.add_argument("-fakecontloss", type=float)
parser.add_argument("-borj", type=float)

args = parser.parse_args()

dict_params = {"faked": "Fake-content D",
                "reald": "Real-content D",
                "sparse": "Sparse",
                "Rb": "R_b",
                "Rt": "R_t",
                "L1image": "L1 image",
                "fakecontloss": "fake_content loss",
                "borj": "borj4_w300"
        }

with open(args.path, "r+") as jsonFile:
    data = json.load(jsonFile)
    for key,value in vars(args).items():
        if value != None and key != "path":
            key_param = dict_params[key]
            data[key_param] = value
            jsonFile.seek(0)
            json.dump(data, jsonFile, indent=2)
            jsonFile.truncate()
