r""" PASCAL-5i few-shot semantic segmentation dataset """

import os
import random

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DatasetGSO(Dataset):
    def __init__(
        self, datapath, fold, transform, split, shot=14, use_original_imgsize=False
    ):
        self.split = "val" if split in ["val", "test"] else "trn"
        self.fold = fold
        self.nfolds = 1
        self.class_ids = [1]  # [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.nclass = len(self.class_ids)
        self.benchmark = "gso"
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        self.rgb_path = "/home/icetenny/senior-1/Datasets_Megapose/Processed/rgb"
        self.mask_path = "/home/icetenny/senior-1/Datasets_Megapose/Processed/mask"
        self.template_path = (
            "/home/icetenny/senior-1/SAM-6D/SAM-6D/Data/gso_obj/templates"
        )
        self.transform = transform

        self.img_metadata = self.build_img_metadata()
        # self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        # idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        # query_name, support_names, class_sample = self.sample_episode(idx)

        query_name = self.img_metadata[idx]
        support_names = [str(i) + ".png" for i in range(self.shot)]

        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = (
            self.load_frame(query_name, support_names)
        )

        query_img = self.transform(query_img)
        if not self.use_original_imgsize:
            query_cmask = F.interpolate(
                query_cmask.unsqueeze(0).unsqueeze(0).float(),
                query_img.size()[-2:],
                mode="nearest",
            ).squeeze()
        # query_mask, query_ignore_idx = self.extract_ignore_idx(
        #     query_cmask.float(), class_sample
        # )
        query_mask = query_cmask.float()

        support_imgs = torch.stack(
            [self.transform(support_img) for support_img in support_imgs]
        )

        support_masks = []
        # support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(
                scmask.unsqueeze(0).unsqueeze(0).float(),
                support_imgs.size()[-2:],
                mode="nearest",
            ).squeeze()
            # support_mask, support_ignore_idx = self.extract_ignore_idx(
            #     scmask, class_sample
            # )
            support_mask = scmask
            support_masks.append(support_mask)
            # support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        # support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {
            "query_img": query_img,
            "query_mask": query_mask,
            "query_name": query_name,
            # "query_ignore_idx": query_ignore_idx,
            "org_query_imsize": org_qry_imsize,
            "support_imgs": support_imgs,
            "support_masks": support_masks,
            "support_names": support_names,
            # "support_ignore_idxs": support_ignore_idxs,
            # "class_id": torch.tensor(self.class_ids),
        }

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):

        # Get obj name and rgb id
        obj_name = query_name.split("/")[0]
        rgb_id = query_name.split("/")[1].split("_")[0]

        # Load Query
        query_img = self.read_rgb(os.path.join(self.rgb_path, rgb_id + ".png"))
        query_mask = self.read_mask(os.path.join(self.mask_path, query_name))

        # Load Supports
        support_imgs = [
            self.read_rgb(os.path.join(self.template_path, obj_name, f"rgb_{name}"))
            for name in support_names
        ]
        support_masks = [
            self.read_mask(os.path.join(self.template_path, obj_name, f"mask_{name}"))
            for name in support_names
        ]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, mask_path):
        r"""Return segmentation mask in PIL Image"""

        image_array = np.array(Image.open(mask_path))

        # Check if the image is 3d, convert to 2d mask
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            binary_mask = torch.tensor(
                np.all(image_array == [255, 255, 255], axis=-1).astype(np.uint8)
            )
        elif len(image_array.shape) == 2:
            binary_mask = torch.tensor(image_array)
        else:
            raise ValueError("Unsupported image format")

        return binary_mask

    def read_rgb(self, rgb_path):
        r"""Return RGB image in PIL Image"""
        return Image.open(rgb_path)

    def build_img_metadata(self):
        """
        return list of image_id
        """

        rgb_img_id = os.listdir(self.rgb_path)
        mask_obj_folder = os.listdir(self.mask_path)

        mask_img_ids = []

        for mask_obj_name in mask_obj_folder:
            obj_img_ids = os.listdir(os.path.join(self.mask_path, mask_obj_name))

            for obj_img_id in obj_img_ids:
                # mask_id = "{obj_name}/{image_id}_{in_image_id}.png"
                mask_id = f"{mask_obj_name}/{obj_img_id}"
                mask_img_ids.append(mask_id)

        print("Total (%s) RGB images are : %d" % (self.split, len(rgb_img_id)))
        print("Total (%s) Mask images are : %d" % (self.split, len(mask_img_ids)))

        return mask_img_ids
