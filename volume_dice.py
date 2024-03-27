import importlib
from collections import defaultdict
import numpy as np
import os
import nibabel as nib

module_name_list = [
    "sam_med3d",
    "sam_med3d_accuracy"
    # "sam_med2d_ft2d"
]
predict_path = '/media/barry/SELF/DeepEyes_Project/20240104/SAM-Med3D/results/vis_sam_med3d'
for module_idx, module_name in enumerate(module_name_list):
    try:
        module = importlib.import_module("results."+module_name)
    except:
        raise ValueError("file not found", module_name)
    if module_name == "sam_med3d":
        dice_Ts = module.dice_Ts
        volume = []
    elif module_name == "sam_med3d_accuracy":
        dice_Ts = module.accuracy_Ts
        volume = []
    tmp_value = []
    for k, v in dice_Ts.items():
        k = k.split("/")
        # print(k)
        cls, dataset, data_type, case = k[-4], k[-3], k[-2], k[-1]
        # print(cls, dataset, data_type, case)
        predict_data = nib.load(os.path.join(predict_path,cls,dataset,case.replace('.nii.gz','_pred0.nii.gz'))).get_fdata()
        volume.append(predict_data.sum())
        tmp_value.append(v*100)
    if module_name == "sam_med3d":
        dice_value = tmp_value
    elif module_name == "sam_med3d_accuracy":
        accuracy_value = tmp_value

vol_dice = np.zeros((len(dice_value),3))
vol_dice[:,0] = np.array(volume)
vol_dice[:,1] = np.array(dice_value)
vol_dice[:,2] = np.array(accuracy_value)

print('finish')