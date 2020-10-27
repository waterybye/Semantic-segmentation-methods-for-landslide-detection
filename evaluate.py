import numpy as np

def weird_division(x, y):
    return 0 if y == 0 else x / y

def modeleval(save_test_model, var, f):
    results = np.load(f'data/results/{save_test_model}.npz')
    inputs, labels, outputs = results['inputs'], results['labels'], results['outputs']
    del results

    n_samples, n_classes, w, h = outputs.shape
    predictions = outputs.argmax(axis=1)

    r_count = 0
    PA, mPA = 0, 0
    m_IOU, background_iou, landslide_iou = 0, 0, 0
    pre, rec, f1 = 0, 0, 0
    for label, pred in zip(labels, predictions):
        PA += (pred == label).sum() / (w * h) #pixel accuracy
        mPA += weird_division(((pred == 0) & (label == 0)).sum(), ((label == 0).sum())) + weird_division(((pred == 1) & (label == 1)).sum(), ((label == 1).sum()))

        background_iou += weird_division(((pred == 0) & (label == 0)).sum(), ((pred == 0) | (label == 0)).sum())
        landslide_iou += weird_division(((pred == 1) & (label == 1)).sum(), ((pred == 1) | (label == 1)).sum())

        pre += weird_division(((pred == 1) & (label == 1)).sum(), (pred == 1).sum())
        rec += weird_division(((pred == 1) & (label == 1)).sum(), (label == 1).sum())

        if weird_division(((pred == 1) & (label == 1)).sum(), (label == 1).sum()) >= 0.3:
            r_count += 1


    PA /= n_samples
    mPA /= (n_samples * 2)
    background_iou /= n_samples
    landslide_iou /= n_samples
    mIOU = (background_iou + landslide_iou) / 2
    pre /= n_samples
    rec /= n_samples
    f1 = (2 * pre * rec) / (pre + rec)

    acc = r_count / n_samples

    f.write('**********' + var + '**********' + '\n')
    f.write(f'PA: {PA:.4f}, mPA: {mPA:.4f}'+'\n')
    f.write(f'mIOU: {mIOU:.4f}, background_iou: {background_iou:.4f}, landslide_iou: {landslide_iou:.4f}'+'\n')
    f.write(f'pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}'+'\n')
    f.write(f'acc: {acc:.4f}'+'\n')