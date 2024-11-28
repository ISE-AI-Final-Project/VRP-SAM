r""" PASCAL-5i few-shot semantic segmentation dataset """

import json
import os
import random

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DatasetGSO_DETR(Dataset):
    def __init__(
        self, datapath, fold, transform, split, shot=14, use_original_imgsize=False
    ):
        self.split = "val" if split in ["val", "test"] else "trn"
        self.fold = fold
        self.nfolds = 1
        self.class_ids = [i for i in range(944)]
        self.nclass = len(self.class_ids)
        self.benchmark = "gso_detr"
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        if datapath == "rpd":
            self.rgb_path = "/workspace/Datasets_Megapose/Processed/rgb"
            self.mask_path = "/workspace/Datasets_Megapose/Processed/mask"
            self.anno_path = "/workspace/Datasets_Megapose/Processed/annotation"
            self.template_path = "/workspace/Datasets_Megapose/gso_obj/templates"
        else:
            self.rgb_path = (
                "/home/icetenny/senior-1/Datasets_Megapose/Processed_DETR/rgb"
            )
            self.mask_path = "/home/icetenny/senior-1/Datasets_Megapose/Processed_DETR/mask"  # Combined original mask
            self.anno_path = (
                "/home/icetenny/senior-1/Datasets_Megapose/Processed_DETR/annotation"
            )
            self.template_path = (
                "/home/icetenny/senior-1/SAM-6D/SAM-6D/Data/gso_obj/templates"
            )

        self.transform = transform

        self.obj_list = []

        self.img_metadata = self.build_img_metadata()
        # self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        # idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        # query_name, support_names, class_sample = self.sample_episode(idx)

        query_id, anno_data = self.img_metadata[idx]

        support_names = [str(i) + ".png" for i in range(self.shot)]

        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = (
            self.load_frame(query_id, support_names, anno_data)
        )

        query_img = self.transform(query_img)

        # if not self.use_original_imgsize:
        #     query_cmask = F.interpolate(
        #         query_cmask.unsqueeze(0).unsqueeze(0).float(),
        #         query_img.size()[-2:],
        #         mode="nearest",
        #     ).squeeze()

        # query_mask = query_cmask.float()

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

        unique_obj_id = []
        bboxes = []
        for instance_anno_data in anno_data:
            unique_obj_id.append(instance_anno_data[0])
            bboxes.append(instance_anno_data[1:])

        batch = {
            "query_img": query_img,
            # "query_mask": query_mask,
            "query_name": query_id,
            "org_query_imsize": org_qry_imsize,
            "support_imgs": support_imgs,
            "support_masks": support_masks,
            "support_names": support_names,
            # "class_id": self.get_class_id(query_name),
            "unique_obj_id": unique_obj_id,
            "bboxes": bboxes,
        }

        return batch

    # def extract_ignore_idx(self, mask, class_id):
    #     boundary = (mask / 255).floor()
    #     mask[mask != class_id + 1] = 0
    #     mask[mask == class_id + 1] = 1

    #     return mask, boundary

    def load_frame(self, query_id, support_names, anno_data):

        # Get obj name and rgb id
        rgb_id, obj_name = query_id.split(":")

        # rgb_id = query_name.split("/")[1].split("_")[0]

        # Load Query
        query_img = self.read_rgb(os.path.join(self.rgb_path, rgb_id + ".png"))
        # query_mask = self.read_mask(os.path.join(self.mask_path, query_name))

        # TODO
        query_mask = torch.zeros([1])

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
            binary_mask = torch.tensor((image_array == 255).astype(np.uint8))
        else:
            raise ValueError("Unsupported image format")

        return binary_mask

    def read_rgb(self, rgb_path):
        r"""Return RGB image in PIL Image"""
        return Image.open(rgb_path)

    def get_class_id(self, query_name):
        obj_name = query_name.split("/")[0]
        class_id = self.obj_list.index(obj_name)
        return torch.tensor(class_id)

    def build_img_metadata(self):
        """
        return list of {image_id}:{obj_name}
        """

        # rgb_img_id = os.listdir(self.rgb_path)
        anno_files = os.listdir(self.anno_path)
        # mask_obj_folder = sorted(os.listdir(self.mask_path))

        query_ids = list()

        for anno_file in anno_files:

            with open(os.path.join(self.anno_path, anno_file), "r") as file:
                anno_data = json.load(file)
            file.close()

            for unique_obj_name, bboxes in anno_data.items():

                image_id = anno_file.rstrip(".json")

                query_id = f"{image_id}:{unique_obj_name}"

                query_ids.append((query_id, bboxes))

        print("Total (%s) Images are : %d" % (self.split, len(anno_files)))
        print("Total (%s) Query are : %d" % (self.split, len(query_ids)))

        return query_ids
