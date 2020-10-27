import os

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pydensecrf import densecrf as dcrf


def image_dcrf(unary: np.ndarray, image: np.ndarray, n_classes: int) -> np.ndarray or None:

    unary = -np.log(unary)
    unary = unary.transpose(2, 1, 0)
    w, h, c = unary.shape
    unary = unary.transpose(2, 0, 1).reshape(n_classes, -1)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(w, h, n_classes)
    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=np.ascontiguousarray(image), compat=4)

    q = d.inference(5)
    return np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)


def decode_segmap(map: np.ndarray) -> Image:
    n_classes = map.max() + 1
    w, h = map.shape

    colors = [np.array([0, 0, 0]), np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])]

    decoded = np.zeros((w, h, 3))
    for label in range(n_classes):
        decoded[map == label] = colors[label]
    return Image.fromarray(decoded.astype(np.int8), 'RGB')


def plot(model_namec):
    plot_save_dir = f'data/plots/results-{model_namec}'
    if os.path.exists(plot_save_dir) == False:
        os.makedirs(plot_save_dir)
    results = np.load(f'data/results/{model_namec}.npz')
    inputs, labels, outputs = results['inputs'], results['labels'], results['outputs']
    del results

    for idx, image in tqdm(enumerate(inputs)):
        image, output, target = image.transpose([1, 2, 0]), outputs[idx], labels[idx]
        normalized_output = np.exp(output) / np.sum(np.exp(output), axis=0)
        image = Image.fromarray(image.astype(np.int8), 'RGB')
        prediction, landslide = np.argmax(output, axis=0), normalized_output[1]

        iw, ih, _ = np.array(image).shape
        n_classes, tw, th = output.shape
        left, top, right, bottom = (iw - tw) / 2, (ih - th) / 2, (iw + tw) / 2, (ih + th) / 2

        draw = ImageDraw.Draw(image)
        draw.rectangle([(left, top), (right, bottom)], outline=(255, 0, 0), width=2)
        del draw

        target_map = decode_segmap(target)
        prediction_map = decode_segmap(prediction)

        landslide = np.stack([landslide * 255] * 3, axis=-1)
        landslide = Image.fromarray(landslide.astype(np.int8), 'RGB')

        image.save(os.path.join(plot_save_dir, f'{idx}-image.bmp'))
        target_map.save(os.path.join(plot_save_dir, f'{idx}-target.bmp'))
        prediction_map.save(os.path.join(plot_save_dir, f'{idx}-prediction.bmp'))
        landslide.save(os.path.join(plot_save_dir, f'{idx}-landslide.bmp'))

        try:
            mask = image_dcrf(normalized_output, np.array(image)[int(left):int(right), int(top):int(bottom)], n_classes)
            mask = decode_segmap(mask)
            mask.save(os.path.join(plot_save_dir, f'{idx}-crf.bmp'))
        except ModuleNotFoundError:
            pass