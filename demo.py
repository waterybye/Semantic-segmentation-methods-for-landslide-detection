import sys
import numpy as np
import torch

from models.deeplabv32 import DeepLabV3
from evaluate import modeleval
from plot import plot
from utils import test_model, get_dataloaders

num_classes = 2
model_name = 'deeplabv3_2_0.001_1.0-2.070'
model_ft = DeepLabV3(num_classes)
on_device = torch.device("cpu")

# test model
checkpoint = torch.load(f'data/models/{model_name}.pth',map_location='cuda:0')
model_ft.load_state_dict(checkpoint['model_state_dict'])
test_List = {'jinsha_test', }

f = open(f'data/evals/{model_name}.txt', 'x')
f.write(model_name + '\n')
for var in test_List:
    save_test_model = model_name + '-' + var
    data_loaders = get_dataloaders(batch_size=2, test=var)
    inputs, labels, outputs = test_model(model_ft, data_loaders['test'], device=on_device)
    np.savez(f'data/results/{save_test_model}.npz', inputs=inputs, labels=labels, outputs=outputs)

    # evaluate the model
    modeleval(save_test_model, var, f)
    # plot test results
    plot(save_test_model)
f.close()
sys.exit(0)