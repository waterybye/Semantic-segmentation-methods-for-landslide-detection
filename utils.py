import copy
import os
import time
import numpy as np
import torch
from torch import nn
from albumentations import (
    CLAHE, RandomRotate90, Transpose, Blur, OpticalDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip,
    OneOf, Compose,
    RandomSizedBBoxSafeCrop)
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from data_utils.dataset import SegmentationSet2, TestSet2


def model_save(path: str, epoch: int, loss, model_state_dict, optimizer_state_dict, **kwargs):
    to_save = dict()
    to_save['epoch'], to_save['loss'], to_save['model_state_dict'], to_save[
        'optimizer_state_dict'] = epoch, loss, model_state_dict, optimizer_state_dict
    for key, val in kwargs.items():
        to_save[key] = val
    ndir, _ = os.path.split(path)
    if not os.path.exists(ndir):
        os.makedirs(ndir)
    torch.save(to_save, path)

def initialize_model(model_name: str, num_classes: int, use_pretrained: bool = True) -> nn.Module:
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg11_bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet1_0":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes

    elif model_name == "densenet121":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception_v3":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        for param in model_ft.parameters():
            param.requires_grad = False
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def get_iou(pred, label):
    # landslide IoU
    landslide_union = np.logical_or(pred == 1, label == 1)
    landslide_intersection = np.logical_and(pred == 1, label == 1)

    landslide_iou = landslide_intersection.sum() / landslide_union.sum() if landslide_union.sum() > 0 else 1
    return landslide_iou


def train_model(model: nn.Module, dataloaders, criterion, optimizer, n_class, scheduler=None, num_epochs=25,
                device=torch.device('cpu')):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_iou = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tp, tn, fp, fn = [[0. for _ in range(n_class)] for _ in range(4)]  # TP, TN, FP, FN for each class

            landslide_union = 0.
            landslide_intersection = 0.

            # Iterate over data.
            for names, inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs.data, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()  # zero the parameter gradients
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                for i in range(n_class):
                    tp[i] += torch.sum((preds == i) & (labels.data == i)).float()
                    tn[i] += torch.sum((preds != i) & (labels.data != i)).float()
                    fp[i] += torch.sum((preds == i) & (labels.data != i)).float()
                    fn[i] += torch.sum((preds != i) & (labels.data == i)).float()

                landslide_intersection += torch.sum((preds == 1) & (labels.data == 1)).float()
                landslide_union += torch.sum((preds == 1) | (labels == 1)).float()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            n_samples = len(dataloaders[phase].dataset) * inputs.size(2) * inputs.size(3)
            epoch_acc = running_corrects.float() / n_samples
            epoch_iou = landslide_intersection / landslide_union

            f1_score = [(2 * tp[i] / (2 * tp[i] + fp[i] + fn[i])).item() for i in range(n_class)]
            accs = [((tp[i] + tn[i]) / n_samples).item() for i in range(n_class)]
            weights = [0.04, 1., 2.38]
            epoch_weighted_acc = sum((accs[i] * weights[i]) / sum(weights) for i in range(n_class))

            print(f'{phase} Loss: {epoch_loss:.4f}')
            print(f'Acc: {epoch_acc:.4f}')
            print(f'Weighted Acc {epoch_weighted_acc:.4f}')
            print(f'F1-scores: {f1_score[0]:.3f}/{f1_score[1]:.3f}/{f1_score[2]:.3f}')
            print(f'Landslide IoU: {epoch_iou:.2f}')
            print()

            if phase == 'train' and scheduler:
                scheduler.step(epoch_loss)

            # update weight as Error Corrective Boosting: w_i = (median(a) - min(a) + smooth) / (a_i - min(a) + smooth)
            if phase == 'val' and hasattr(criterion, 'update_weight'):
                mediana, mina = np.median(f1_score), np.min(f1_score)
                new_weights = torch.zeros(n_class, dtype=torch.float32)
                for i in range(new_weights.size()[0]):
                    new_weights[i] = (mediana - mina + .5) / (f1_score[i] - mina + .5)
                criterion.update_weight(new_weights.to(device))

            # deep copy the model
            if phase == 'val' and epoch_weighted_acc > best_acc:
                best_acc = epoch_weighted_acc
            if phase == 'val' and epoch_iou >= best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    print(f'Best val landslide IoU: {best_iou:.5f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, dataloader, device=torch.device('cpu')):
    since = time.time()
    inputs_list, outputs_list, labels_list = list(), list(), list()

    model.eval()  # Set model to evaluate mode

    # Iterate over data.
    for names, inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        inputs_list.append(inputs.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs_list.append(outputs.cpu().numpy())

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return np.concatenate(inputs_list), np.concatenate(labels_list), np.concatenate(outputs_list)


def pretrain_resnet(model, model_name, learning_rate, dataloaders, num_epochs=25, device=torch.device('cpu'),
                    n_class=43):
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    from torch import optim
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=learning_rate)
    # optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=2, verbose=True)
    # scheduler = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=(torch.tensor([1.9413, 2.1518, 1.8227, 2.4867, 2.6754, 2.9754,
                                                               3.9839, 4.2460, 7.9527, 6.1611, 6.8814, 15.9447,
                                                               30.1597, 37.2683, 42.7707, 45.6151, 40.5372, 45.2004,
                                                               47.2388, 52.0371, 53.5295, 72.9905, 93.1514, 101.3732,
                                                               103.2222, 110.1599, 121.9205, 135.9204, 157.8678,
                                                               178.4537, 313.5603, 337.1113, 431.0624, 386.8559,
                                                               346.3839, 479.4246, 487.1006, 538.8535, 632.3403,
                                                               630.7993, 1145.4437, 1223.8561, 1707.3519],
                                                              device=device)))

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    try:

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                tp, tn, fp, fn = [[0. for _ in range(n_class)] for _ in range(4)]  # TP, TN, FP, FN for each class

                # Iterate over data.
                for names, inputs, labels in tqdm(dataloaders[phase]):
                    inputs, labels = inputs.to(device), labels.to(device)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)

                        loss = criterion(outputs, labels)

                        preds = outputs.data
                        preds[preds >= 0.] = 1.
                        preds[preds < 0.] = 0.
                        # _, preds = torch.max(outputs.data, 1) #

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            optimizer.zero_grad()  # zero the parameter gradients
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    for i in range(n_class):
                        tp[i] += torch.sum((preds[:, i] == 1) & (labels.data[:, i] == 1)).float()
                        tn[i] += torch.sum((preds[:, i] == 0) & (labels.data[:, i] == 0)).float()
                        fp[i] += torch.sum((preds[:, i] == 1) & (labels.data[:, i] == 0)).float()
                        fn[i] += torch.sum((preds[:, i] == 0) & (labels.data[:, i] == 1)).float()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                n_samples = len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.float() / n_samples / n_class

                f1_score = [(2 * tp[i] / (2 * tp[i] + fp[i] + fn[i])).item() for i in range(n_class)]
                accs = [((tp[i] + tn[i]) / n_samples).item() for i in range(n_class)]

                print(f'{phase} Loss: {epoch_loss:.4f}')
                print(f'Acc: {epoch_acc:.4f}')
                print('Accs: ' + '/'.join([f'{accs[i] * 100:.2f}%' for i in range(n_class)]))
                print('F1-scores: ' + '/'.join([f'{f1_score[i]:.3f}' for i in range(n_class)]))
                print()

                if phase == 'train' and scheduler:
                    scheduler.step(epoch_loss)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                if phase == 'val':
                    val_acc_history.append(epoch_loss)
            model_save(f'data/models/{model_name}_{epoch:03}.pth', num_epochs, val_acc_history[-1], model.state_dict(),
                       optimizer.state_dict())
    finally:
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        model_save(f'data/models/{model_name}.pth', num_epochs, val_acc_history[-1], model.state_dict(),
                   optimizer.state_dict())
        exit(0)
    return model


def get_dataloaders(input_size=(420, 420), batch_size=2, test=None ):
    pnghigh_img_dir, pnghigh_ann_dir = r'data/npz/pnghigh/image', r'data/npz/pnghigh/mask' #56
    pnglow_img_dir, pnglow_ann_dir = r'data/npz/pnglow/image', r'data/npz/pnglow/mask'  # 12
    pngother_img_dir,pngother_ann_dir = r'data/npz/pngother/image', r'data/npz/pngother/mask' #22

    pnghigh_img_lists = np.array(list(map(lambda x: os.path.join(pnghigh_img_dir, x), sorted(os.listdir(pnghigh_img_dir)))))
    pnghigh_ann_lists = np.array(list(map(lambda x: os.path.join(pnghigh_ann_dir, x), sorted(os.listdir(pnghigh_ann_dir)))))

    pnglow_img_lists = np.array(list(map(lambda x: os.path.join(pnglow_img_dir, x), sorted(os.listdir(pnglow_img_dir)))))
    pnglow_ann_lists = np.array(list(map(lambda x: os.path.join(pnglow_ann_dir, x), sorted(os.listdir(pnglow_ann_dir)))))

    pngother_img_lists = np.array(list(map(lambda x: os.path.join(pngother_img_dir, x), sorted(os.listdir(pngother_img_dir)))))
    pngother_ann_lists = np.array(list(map(lambda x: os.path.join(pngother_ann_dir, x), sorted(os.listdir(pngother_ann_dir)))))

    indices_high = np.arange(len(pnghigh_img_lists))
    #np.random.shuffle(indices_high)
    #trn1, val1, tst1 = np.split(indices1, [int(0.7 * len(pngs2_image_lists)), int(0.8 * len(pngs2_image_lists))])
    trn_high, val_high, tst_high = np.split(indices_high, [42, 48])

    indices_low = np.arange(len(pnglow_img_lists))
    #np.random.shuffle(indices_low)
    tst_low = indices_low

    indices_other = np.arange(len(pngother_img_lists))
    #np.random.shuffle(indices_other)
    # trn1, val1, tst1 = np.split(indices1, [int(0.7 * len(pngs2_image_lists)), int(0.8 * len(pngs2_image_lists))])
    trn_other, val_other, tst_other = np.split(indices_other, [12, 16])


    transform = Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OpticalDistortion(p=0.3),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        RandomSizedBBoxSafeCrop(*input_size, erosion_rate=0.3)
    ])

    np_trn_img = np.hstack((pnghigh_img_lists[trn_high], pngother_img_lists[trn_other]))
    np_trn_ann = np.hstack((pnghigh_ann_lists[trn_high], pngother_ann_lists[trn_other]))

    np_val_img = np.hstack((pnghigh_img_lists[val_high], pngother_img_lists[val_other]))
    np_val_ann = np.hstack((pnghigh_ann_lists[val_high], pngother_ann_lists[val_other]))

    if test is 'all_test':
        np_tst_img = np.hstack((pnghigh_img_lists[tst_high], np.hstack((pnglow_img_lists[tst_low], pngother_img_lists[tst_other]))))
        np_tst_ann = np.hstack((pnghigh_ann_lists[tst_high], np.hstack((pnglow_ann_lists[tst_low], pngother_ann_lists[tst_other]))))
    elif test is 'jinsha_test':
        np_tst_img = np.hstack((pnghigh_img_lists[tst_high],pnghigh_img_lists[tst_low]))
        np_tst_ann = np.hstack((pnghigh_ann_lists[tst_high],pnghigh_ann_lists[tst_low]))
    elif test is 'other_test':
        np_tst_img = pnghigh_img_lists[tst_other]
        np_tst_ann = pnghigh_ann_lists[tst_other]
    elif test is 'jinsha_train':
        np_tst_img = pnghigh_img_lists[trn_high]
        np_tst_ann = pnghigh_ann_lists[trn_high]
    elif test is 'other_train':
        np_tst_img = pnghigh_img_lists[trn_other]
        np_tst_ann = pnghigh_ann_lists[trn_other]
    elif test is 'jinsha_val':
        np_tst_img = pnghigh_img_lists[val_high]
        np_tst_ann = pnghigh_ann_lists[val_high]
    else:
        np_tst_img = pnghigh_img_lists[val_other]
        np_tst_ann = pnghigh_ann_lists[val_other]

    image_datasets = {
        'train': SegmentationSet2(np_trn_img,np_trn_ann, transform=transform),
        'val': SegmentationSet2(np_val_img,np_val_ann, transform=transform),
        'test': TestSet2(np_tst_img,np_tst_ann)
    }

    data_loaders = {
        phase: DataLoader(image_datasets[phase], batch_size=batch_size, shuffle=(phase in ['train']), drop_last=False)
        for phase in ['train', 'val', 'test']}

    return data_loaders
