r""" Dataloader builder for few-shot semantic segmentation dataset  """

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.coco import DatasetCOCO
from data.gso import DatasetGSO
from data.gso_detr import DatasetGSO_DETR
from data.linemod import DatasetLINEMOD
from data.pascal import DatasetPASCAL

# from data.coco2pascal import DatasetCOCO2PASCAL


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):

        cls.datasets = {
            "pascal": DatasetPASCAL,
            "coco": DatasetCOCO,
            # 'coco2pascal': DatasetCOCO2PASCAL,
            "linemod": DatasetLINEMOD,
            "gso": DatasetGSO,
            "gso_detr": DatasetGSO_DETR,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cls.transform = transforms.Compose(
            [
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(cls.img_mean, cls.img_std),
            ]
        )

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == "trn"
        nworker = nworker if split == "trn" else 0

        dataset = cls.datasets[benchmark](
            cls.datapath,
            fold=fold,
            transform=cls.transform,
            split=split,
            shot=shot,
            use_original_imgsize=cls.use_original_imgsize,
        )
        if split == "trn":
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle
            )
            shuffle = False
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=shuffle
            )
            pin_memory = True

        if benchmark == "gso_detr":
            # Custom collate fn for gso_detr

            dataloader = DataLoader(
                dataset,
                batch_size=bsz,
                shuffle=False,
                pin_memory=True,
                num_workers=nworker,
                sampler=sampler,
                collate_fn=cls.gso_detr_collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=bsz,
                shuffle=False,
                pin_memory=True,
                num_workers=nworker,
                sampler=sampler,
            )

        return dataloader

    @staticmethod
    def gso_detr_collate_fn(batch):
        """
        Custom collate function to handle variable-sized fields in a batch.

        Args:
            batch: List of dictionaries, where each dictionary represents a single data sample.

        Returns:
            dict: Batched data with images, annotations, and variable-sized fields handled properly.
        """
        # Initialize output dictionary
        collated_batch = {
            "query_img": torch.stack([item["query_img"] for item in batch]),
            "query_name": [item["query_name"] for item in batch],
            "org_query_imsize": [item["org_query_imsize"] for item in batch],
            "support_imgs": torch.stack([item["support_imgs"] for item in batch]),
            "support_masks": torch.stack([item["support_masks"] for item in batch]),
            "support_names": [item["support_names"] for item in batch],
            "unique_obj_id": [item["unique_obj_id"] for item in batch],
            "bboxes": [item["bboxes"] for item in batch],
        }

        return collated_batch
