from typing import Tuple
import numpy as np
import cv2
from ..utils import ImageUtil
from ..augmentation import segmentation_aug, classification_aug

import torch.utils.data as data
from torchvision import transforms

class SegmentationDataset:
    """for segmentation task
    read image and mask, apply augmentation
    """

    def __init__(
            self,
            annotations: list,
            input_shape: Tuple[int, int],
            nb_classes: int,
            mean: np.ndarray = np.zeros(3),
            std: np.ndarray = np.ones(3),
            augmentation: list = list(),
            augmix: bool = False,
            train_colors: list = list(),
            **kwargs
    ):

        self.annotations = annotations
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        self.augmix = augmix
        self.train_colors = train_colors

    def __getitem__(self, i):

        # input_file is [image_path]
        # label is mask_image_path
        *input_file, label = self.annotations[i]
        # read images
        input_image = self.image_util.read_image(input_file[0])
        label = self.image_util.read_image(label, self.train_colors)

        # apply augmentations
        if self.augmentation and len(self.augmentation) > 0:
            input_image, label = segmentation_aug(
                input_image, label,
                self.mean, self.std,
                self.augmentation,
                self.augmix
            )

        # apply preprocessing
        # resize
        input_image = self.image_util.resize(input_image, anti_alias=True)
        label = self.image_util.resize(label, anti_alias=False)
        # normalization and onehot encoding
        input_image = self.image_util.normalization(input_image)
        label = self.image_util.cast_to_onehot(label)

        return input_image, label

    def __len__(self):
        return len(self.annotations)


class ClassificationDataset:
    """for classification
    read image/frame of video and class id
    """

    def __init__(
            self,
            annotations: list,
            input_shape: Tuple[int, int],
            nb_classes: int,
            mean: list,
            std: list,
            augmentation: list = list(),
            input_data_type: str = "image",
            **kwargs
    ):

        self.annotations = annotations
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.augmentation = augmentation
        self.input_data_type = input_data_type
        self.mean = mean
        self.std = std

    def __getitem__(self, i):

        # input_file is [image_path] or [video_path, frame_id]
        # label is class_id
        *input_file, label = self.annotations[i]

        if self.input_data_type == "video":
            # video data [video_path, frame_id]
            video_path, frame_id = input_file
            # read frame
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, input_image = video.read()
            # BGR -> RGB
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

            # 背景の黒を消す
            _width = input_image.shape[1]
            width_to_remove = round(_width * 0.05)
            input_image = input_image[:, width_to_remove:-width_to_remove]

        elif self.input_data_type == "image":
            # image data [image_path]
            input_image = self.image_util.read_image(input_file[0])

        # apply preprocessing
        input_image = self.image_util.resize(input_image, anti_alias=True)
        input_image = self.image_util.normalization(input_image)
        label = self.image_util.cast_to_onehot(label)

        # 平均を引いて標準偏差で割る
        if self.mean and self.std:
            input_image = (input_image - self.mean) / self.std

        # apply augmentations
        if self.augmentation and len(self.augmentation) > 0:
            input_image, label = classification_aug(
                input_image, label,
                self.mean, self.std,
                self.augmentation,
                self.augmix
            )

        return input_image, label

    def __len__(self):
        return len(self.annotations)

# 入力画像の前処理をするクラス
# 訓練時と推論時で処理が異なる


class ImageTransform():
    """
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする。


    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)

class pytorchClassificationDataset(data.Dataset):
    """
    PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    """

    def __init__(
        self,
        annotations: list,
        input_shape: Tuple[int, int],
        nb_classes: int,
        # augmentation: list = list(),
        input_data_type: str = "image",
        transform=None,
        phase='train',
        **kwargs
    ):
        self.annotations = annotations
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        # self.augmentation = augmentation
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.annotations)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # input_file is [image_path] or [video_path, frame_id]
        # label is class_id
        *input_file, label = self.annotations[i]

        if self.input_data_type == "video":
            # video data [video_path, frame_id]
            video_path, frame_id = input_file
            # read frame
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, input_image = video.read()
            # BGR -> RGB
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        elif self.input_data_type == "image":
            # image data [image_path]
            input_image = self.image_util.read_image(input_file[0])

        # apply preprocessing
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])
        label = self.image_util.cast_to_onehot(label)

        return img_transformed, label
