import os.path as osp
import os
import json
import shutil
import nibabel as nib
from tqdm import tqdm
import torchio as tio

dataset_root = "/media/barry/SELF/DeepEyes_Project/nnUNetFrame/DATASET/nnUNet_raw"
dataset_list = [
    # 'Dataset501_AMOS22',
    # 'Dataset502_WORD',
    'Dataset505_TotalSegmentatorv2All'
    # 'Dataset506_TotalSegmentatorv2Compress'
]

target_dir = "/media/barry/SELF/DeepEyes_Project/medical_preprocessed/sam_med_3d/validation_all"#sam_med_3d


for dataset in dataset_list:
    dataset_dir = osp.join(dataset_root, dataset)
    meta_info = json.load(open(osp.join(dataset_dir, "dataset.json")))

    print(meta_info['name'], meta_info['channel_names'])
    num_classes = len(meta_info["labels"])-1
    print("num_classes:", num_classes, meta_info["labels"])
    for cls_name, idx in meta_info["labels"].items():
        cls_name = cls_name.replace(" ", "_")
        idx = int(idx)
        if(idx<1): continue
        dataset_name = dataset.split("_", maxsplit=1)[1]
        target_cls_dir = osp.join(target_dir, cls_name, dataset_name)
        target_img_dir = osp.join(target_cls_dir, "imagesTs")#imagesTr
        target_gt_dir = osp.join(target_cls_dir, "labelsTs")#labelsTr
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_gt_dir, exist_ok=True)
        
        source_cls_dir = osp.join(dataset_root, dataset)
        source_img_dir = osp.join(source_cls_dir, "imagesTs")#imagesTr
        source_gt_dir = osp.join(source_cls_dir, "labelsTs")#labelsTr
        for item in tqdm(os.listdir(source_img_dir), desc=f"{dataset_name}-{cls_name}"):
            img = osp.join(source_img_dir, item)
            gt = osp.join(source_gt_dir, item.replace("_0000.nii.gz", ".nii.gz"))

            target_img_path = osp.join(target_img_dir, osp.basename(img).replace("_0000.nii.gz", ".nii.gz"))
            # print(f"{img}->{target_img}")
            target_gt_path = osp.join(target_gt_dir, osp.basename(gt))
            # print(f"{gt}->{target_gt_path}")

            gt_img = nib.load(gt)    
            spacing = tuple(gt_img.header['pixdim'][1:4])
            # Calculate the product of the spacing values
            spacing_voxel = spacing[0] * spacing[1] * spacing[2]
            gt_arr = gt_img.get_fdata()
            gt_arr[gt_arr != idx] = 0
            gt_arr[gt_arr != 0] = 1*idx
            volume = gt_arr.sum()*spacing_voxel
            # print("volume:", volume)
            if(volume<10): 
                print("skip", target_img_path)
                continue
            shutil.copy(img, target_img_path)
            new_gt_img = nib.Nifti1Image(gt_arr, gt_img.affine, gt_img.header)
            new_gt_img.to_filename(target_gt_path)