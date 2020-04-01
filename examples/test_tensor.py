from models.munit.enc_dec import ContentTensor
import torch
import sys
import os
sys.path.append(os.path.join(sys.path[0], '../gans_pytorch/stylegan2'))

tens = torch.ones(4,5,5)
print(type(ContentTensor(tens) + ContentTensor(tens)))
print(ContentTensor(tens) + ContentTensor(tens))
