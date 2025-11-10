import os
import numpy as np
from PIL import Image
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion

from multiprocessing import Pool


def load_and_process_image(args):
    file_path, target_size = args
    return np.array(
        Image.open(file_path).convert('L').resize(target_size),
        dtype=np.uint8
    )


def load_volume_multiprocess(folder_path, volume_list, num_workers=4):
    file_paths = [os.path.join(folder_path, img) for img in volume_list]
    args_list = [(path, (512, 512)) for path in file_paths]

    with Pool(num_workers) as pool:
        results = pool.map(load_and_process_image, args_list)

    return np.stack(results)


def calculate_metrics(pred_3d, gt_3d):
    # 确保是二值图像
    pred_binary = (pred_3d > 0).astype(np.uint8)
    gt_binary = (gt_3d > 0).astype(np.uint8)

    # Dice Score
    intersection = np.sum(pred_binary * gt_binary)
    dice = (2. * intersection) / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-8)

    # Hausdorff Distance 95%
    def hd95(pred, gt):
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            return np.nan

        # 计算距离变换
        pred_dist = distance_transform_edt(1 - pred)
        gt_dist = distance_transform_edt(1 - gt)

        # 获取表面点
        pred_surface = pred * gt_dist
        gt_surface = gt * pred_dist

        # 计算HD95
        hd95_value = np.percentile(np.concatenate([pred_surface[pred_surface > 0],
                                                   gt_surface[gt_surface > 0]]), 95)
        return hd95_value

    # Normalized Surface Dice (NSD)
    def nsd(pred, gt, tolerance=2.0):
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            return 0.0

        # 计算表面点
        pred_surface = pred - binary_erosion(pred)
        gt_surface = gt - binary_erosion(gt)

        # 计算距离变换
        pred_dist = distance_transform_edt(1 - pred)
        gt_dist = distance_transform_edt(1 - gt)

        # 计算NSD
        surface_intersection = np.sum((pred_surface > 0) & (gt_dist <= tolerance)) + \
                               np.sum((gt_surface > 0) & (pred_dist <= tolerance))
        surface_union = np.sum(pred_surface) + np.sum(gt_surface)

        return surface_intersection / (surface_union + 1e-8)

    # 计算3D指标
    hd95_value = hd95(pred_binary, gt_binary)
    nsd_value = nsd(pred_binary, gt_binary)

    return {
        'dice': dice,
        'hd95': hd95_value,
        'nsd': nsd_value
    }


if __name__=='__main__':

    pred_path = 'Your/pred_path'
    gt_path = 'Your/data_path'

    pred_ls = os.listdir(pred_path)
    gt_ls = os.listdir(gt_path)
    pred_ls.sort(), gt_ls.sort()
    print(len(pred_ls), len(gt_ls))

    results_dict = {}

    # get label name
    label_names = []
    for name in pred_ls:
        label_name = name.split('_')[-1][:-4]
        if label_name not in label_names:
            label_names.append(label_name)

    # initialize dict
    for label_name in label_names:
        results_dict[label_name] = []

    # append volume name in dict
    for name in pred_ls:
        volume_name = '-'.join(name.split('-', 2)[:2])

        for key in results_dict.keys():
            if volume_name not in results_dict[key]:
                results_dict[key].append(volume_name)

    print(results_dict)

    overall_dice = 0
    overall_nsd = 0
    overall_hd95 = 0
    num_of_classes = 0

    for key in results_dict.keys():
        num_of_classes += 1
        class_dice = 0
        class_nsd = 0
        class_hd95 = 0
        num_of_volumes = 0

        for volume_name in results_dict[key]:
            num_of_volumes += 1
            volume_list = []
            for name in pred_ls:
                if '-'.join(name.split('-', 2)[:2]) == volume_name and name.split('_')[-1][:-4] == key and name in gt_ls:
                    volume_list.append(name)

            # print(volume_name, volume_list)

            from PIL import Image

            pred_3d = load_volume_multiprocess(pred_path, volume_list, num_workers=8)
            gt_3d = load_volume_multiprocess(gt_path, volume_list, num_workers=8)

            print(pred_3d.shape, gt_3d.shape)

            metrics = calculate_metrics(pred_3d, gt_3d)
            print(f"Dice: {metrics['dice']:.4f}")
            print(f"HD95: {metrics['hd95']:.4f}")
            print(f"NSD: {metrics['nsd']:.4f}")

            class_dice += metrics['dice']
            class_nsd += metrics['nsd']
            class_hd95 += metrics['hd95']

        class_dice = class_dice / num_of_volumes
        class_nsd = class_nsd / num_of_volumes
        class_hd95 = class_hd95 / num_of_volumes
        print('label, class_dice, class_nsd, class_hd95', key, class_dice, class_nsd, class_hd95)
        print('\n')

        overall_dice += class_dice
        overall_nsd += class_nsd
        overall_hd95 += class_hd95

    overall_dice = overall_dice / num_of_classes
    overall_nsd = overall_nsd / num_of_classes
    overall_hd95 = overall_hd95 / num_of_classes
    print('dataset, overall_dice, overall_nsd, overall_hd95', pred_path, overall_dice, overall_nsd, overall_hd95)












