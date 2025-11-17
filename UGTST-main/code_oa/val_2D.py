import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy import ndimage

def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    if pred.sum() == 0 and gt.sum() > 0:
        return 0.0, 0.0
    return 0.0, 0.0

def get_largest_component(image):
    dim = len(image.shape)
    if(image.sum() == 0 ):
        # print('the largest component is null')
        return image
    if(dim == 2):
        s = ndimage.generate_binary_structure(2,1)
    elif(dim == 3):
        s = ndimage.generate_binary_structure(3,1)
    else:
        raise ValueError("the dimension number should be 2 or 3")
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    max_label = np.where(sizes == sizes.max())[0] + 1
    output = np.asarray(labeled_array == max_label, np.uint8)
    return output

def _prepare_volume(tensor):
    array = tensor.squeeze(0).cpu().detach().numpy()
    if array.ndim == 4:
        array = array.squeeze(0)
    if array.ndim != 3:
        raise RuntimeError(f"Expected 3D volume, got shape {array.shape}")
    return np.moveaxis(array, -1, 0)


def test_single_volume(image, label, net, classes, patch_size=[128, 128]):
    image = _prepare_volume(image)
    label = _prepare_volume(label)
    label[label < 0] = 0
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice_ = image[ind, :, :]
        x, y = slice_.shape[0], slice_.shape[1]
        slice_resized = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=1)
        input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            logits = net(input_tensor)[0]
            out = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_fast(image, label, net, classes):
    image = _prepare_volume(image)
    label = _prepare_volume(label)
    label[label < 0] = 0
    input = torch.from_numpy(image).float().cuda().unsqueeze(1)
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(
                net(input)[0], dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        prediction = out
    metric_list = []
    for i in range(classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[128, 128]):
    image = _prepare_volume(image)
    label = _prepare_volume(label)
    label[label < 0] = 0
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice_ = image[ind, :, :]
        x, y = slice_.shape[0], slice_.shape[1]
        slice_resized = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=1)
        input_tensor = torch.from_numpy(slice_resized).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input_tensor)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
