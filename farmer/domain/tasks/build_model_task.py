import segmentation_models
from segmentation_models import Unet, PSPNet
from segmentation_models import metrics

from farmer.ncc.models import xception, mobilenet, Deeplabv3, Model2D
from farmer.ncc.losses import loss_functions 
from ..model.task_model import Task

from tensorflow import keras


class BuildModelTask:
    def __init__(self, config):
        self.config = config

    def command(self, trial=None):
        # return: base_model is saved when training on multi gpu

        base_model = self._do_make_model_task(
            task=self.config.task,
            model_name=self.config.model_name,
            nb_classes=self.config.nb_classes,
            height=self.config.height,
            width=self.config.width,
            backbone=self.config.backbone,
            activation=self.config.activation,
            trial=trial
        )
        base_model = self._do_load_model_task(
            base_model, self.config.trained_model_path
        )
        model = self._do_multi_gpu_task(
            base_model, self.config.multi_gpu, self.config.nb_gpu
        )
        compiled_model = self._do_compile_model_task(
            model,
            self.config.optimizer,
            self.config.learning_rate,
            self.config.task,
            self.config.loss,
            trial
        )

        return compiled_model, base_model

    def _do_make_model_task(
        self,
        task,
        model_name,
        nb_classes,
        width=299,
        height=299,
        backbone="resnet50",
        activation="softmax",
        trial=None
    ):
        if task == Task.CLASSIFICATION:
            xception_shape_condition = height >= 71 and width >= 71
            mobilenet_shape_condition = height >= 32 and width >= 32

            if model_name == "xception" and xception_shape_condition:
                model = xception(nb_classes, height, width)
            elif model_name == "mobilenet" and mobilenet_shape_condition:
                model = mobilenet(nb_classes, height, width)
            else:
                model = Model2D(nb_classes, height, width)

        elif task == Task.SEMANTIC_SEGMENTATION:
            print('------------------')
            print('Model:', model_name)
            print('Backbone:', backbone)
            print('------------------')

            if model_name == "unet":
                model = Unet(
                    backbone_name=backbone,
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                )
            elif model_name == "deeplab_v3":
                model = Deeplabv3(
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                    backbone=backbone,
                    activation=activation
                )
            elif model_name == "pspnet":
                model = PSPNet(
                    backbone_name=backbone,
                    input_shape=(height, width, 3),
                    classes=nb_classes,
                )
        else:
            raise NotImplementedError

        return model

    def _do_load_model_task(self, model, trained_model_path):
        if trained_model_path:
            model.load_weights(trained_model_path)
        return model

    def _do_multi_gpu_task(self, base_model, multi_gpu, nb_gpu):
        if multi_gpu:
            if self.config.framework == "tensorflow":
                model = keras.utils.multi_gpu_model(base_model, gpus=nb_gpu)
        else:
            model = base_model
        return model

    def _do_compile_model_task(
        self,
        model,
        optimizer,
        learning_rate,
        task_id,
        loss_func,
        trial
    ):
        if self.config.framework == "tensorflow":
                print('------------------')
                print('Optimizer:', optimizer)
                print('------------------')
            elif optimizer == "adam":
                optimizer = keras.optimizers.Adam(
                    lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.001
                )
            else:
                optimizer = keras.optimizers.SGD(
                    lr=learning_rate, momentum=0.9, decay=0.001
                )

            if task_id == Task.CLASSIFICATION:
                model.compile(
                    optimizer=optimizer,
                    loss=keras.losses.categorical_crossentropy,
                    metrics=["acc"],
                )
            elif task_id == Task.SEMANTIC_SEGMENTATION:
                print('------------------')
                print('Loss:', loss_func)
                print('------------------')
                loss = getattr(loss_functions, loss_func)(
                    **self.config.loss_params
                )
                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=[metrics.iou_score,
                             loss_functions.categorical_crossentropy()],
                )
            else:
                raise NotImplementedError

        return model
