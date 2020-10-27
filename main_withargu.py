import argparse
import sys

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluate import modeleval
from models.deeplab3plus import DeepLabv3_plus
from models.deeplabv32 import DeepLabV3
from models.gcn import GCN
from models.padded_unet import UNet
from models.pspnet import pspnet
from models.fcn import fcn8s
from plot import plot
from utils import train_model, test_model, model_save, get_dataloaders


def get_model(_args):
    if _args.model == 'unet':
        return UNet(_args.channel_in, _args.channel_out)
    elif _args.model == 'fcn':
        return fcn8s(n_classes=_args.num_classes)
    elif _args.model == 'pspnet':
        return pspnet(version='landslide')
    elif _args.model == 'deeplabv3':
        return DeepLabV3(_args.num_classes)
    elif _args.model == 'deeplabv3plus':
        return DeepLabv3_plus(n_classes=_args.num_classes)
    elif _args.model == 'gcn':
        return GCN(_args.num_classes)

def do_train(_model_ft, _model_name, _learning_rate, _class_weights, _data_loaders, _n_epochs, _on_device):
    params_to_update = []
    for name, param in _model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=_learning_rate)
    # optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)

    scheduler = ReduceLROnPlateau(optimizer_ft, mode='min', factor=.5, patience=5, verbose=True)
    # scheduler = None

    criterion = nn.CrossEntropyLoss(_class_weights.to(_on_device))
    # criterion = ECBCrossEntropyLoss(_class_weights.to(_on_device))
    # criterion = JointLoss(_class_weights.to(_on_device))
    # criterion = SoftDiceLoss(_class_weights.to(_on_device))

    _model_ft, hist = train_model(_model_ft, _data_loaders, criterion, optimizer_ft, 3, scheduler=scheduler,
                                  num_epochs=_n_epochs, device=_on_device)

    model_save(f'data/models/{_model_name}.pth', _n_epochs, hist[-1], _model_ft.state_dict(), optimizer_ft.state_dict())
    return _model_ft


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='all', type=str,
                        dest='mode', help='train, test, or plot')
    parser.add_argument('--dataset', default='npz', type=str,
                        dest='dataset', help='npz or others')
    parser.add_argument('--model', default='unet', type=str,
                        dest='model', help='Using which network')
    parser.add_argument('--max_epochs', default=70, type=int,
                        dest='max_epochs', help='Max training epochs')
    parser.add_argument('--channel_in', default=3, type=int,
                        dest='channel_in', help='Number of input channels')
    parser.add_argument('--channel_out', default=2, type=int,
                        dest='channel_out', help='Number of output channels')
    parser.add_argument('--num_classes', default=2, type=int,
                        dest='num_classes', help='Number of class')
    parser.add_argument('--lr', default=1e-4, type=float,
                        dest='learning_rate', help='Learning rate')

    # LOC               29477725, 1352863, 568612 -> 0.04, 1., 2.38
    # RANDOM            27124146, 2130826, 1615028 -> 0.08, 1., 1.32
    # RANDOM-POSITIVE   9901762, 2132390, 313848 -> 0.21, 1., 6.79
    parser.add_argument('--class_weights', default=None, type=float, nargs=2,
                        dest='class_weights', help='Class weights')
    parser.add_argument('--cuda', default=1, type=int,
                        dest='cuda', help='GPU number, in [0, 1, 2, 3]')
    parser.add_argument('--bs', default=2, type=int,
                        dest='batch_size', help='Batch size')
    parser.add_argument('--set_name', default='random', type=str,
                        dest='set_name', help='Data set name')
    parser.add_argument('--test', default='all_test', type=str,
                        dest='test', help='test_data')

    args = parser.parse_args()
    if args.mode is None or args.mode not in ['train', 'test', 'plot', 'eval', 'all']:
        sys.exit(0)

    n_epochs = args.max_epochs
    learning_rate = args.learning_rate
    momentum = .9
    class_weights = torch.tensor(args.class_weights, dtype=torch.float32)

    model_ft = get_model(args)
    model_name = args.model + '_' + str(args.batch_size) + f'_{learning_rate}_' + '-'.join(
        [str(round(float(i) + 0.001, 2)) for i in class_weights]) \
                 + str(n_epochs)

    '''
    model_name = args.model + f'_{learning_rate}_' + '-'.join(
        [str(round(float(i) + 0.001, 2)) for i in class_weights]) \
                 + str(n_epochs)
    '''

    on_device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(on_device)
    data_loaders = get_dataloaders(batch_size=args.batch_size, test=args.test)

    if args.mode == 'train':
        do_train(model_ft, model_name, learning_rate, class_weights, data_loaders, n_epochs, on_device)
        sys.exit(0)
    elif args.mode == 'test':
        checkpoint = torch.load(f'data/models/{model_name}.pth')
        model_ft.load_state_dict(checkpoint['model_state_dict'])

        inputs, labels, outputs = test_model(model_ft, data_loaders['test'], device=on_device)

        np.savez(f'data/results/{model_name}.npz', inputs=inputs, labels=labels, outputs=outputs)
        sys.exit(0)
    elif args.mode == 'plot':
        print(model_name)
        plot(model_name)
    elif args.mode == 'eval':
        modeleval(model_name)
    elif args.mode == 'all':
        #train model
        model_ft = do_train(model_ft, model_name, learning_rate, class_weights, data_loaders, n_epochs, on_device)

        # test model
        checkpoint = torch.load(f'data/models/{model_name}.pth')
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        test_List = {'all_test', 'jinsha_test', 'other_test', 'jinsha_train', 'other_train', 'jinsha_val',
                     'other_val'}

        f = open(f'data/evals/{model_name}.txt', 'x')
        f.write(model_name + '\n')
        for var in test_List:
            save_test_model = model_name + '-' + var
            data_loaders = get_dataloaders(batch_size=args.batch_size, test=var)
            inputs, labels, outputs = test_model(model_ft, data_loaders['test'], device=on_device)
            np.savez(f'data/results/{save_test_model}.npz', inputs=inputs, labels=labels, outputs=outputs)

            # evaluate the model
            modeleval(save_test_model, var, f)
            # plot test results
            plot(save_test_model)
        f.close()
        sys.exit(0)
    else:
        print("no chosen mode, exit")
        sys.exit(0)
