import tensorflow as tf
import segmentation_models
from segmentation_models.base import Loss
from segmentation_models.losses import CategoricalCELoss
from ..losses import functional as F

segmentation_models.set_framework('tf.keras')


class DiceLoss(Loss):
    def __init__(self, beta=1, class_weights=None):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1

    def __call__(self, gt, pr):
        return F.dice_loss(
            gt=gt,
            pr=pr,
            beta=self.beta,
            class_weights=self.class_weights
        )


class JaccardLoss(Loss):
    def __init__(self, class_weights=None):
        super().__init__(name='jaccard_loss')
        self.class_weights = class_weights if class_weights is not None else 1

    def __call__(self, gt, pr):
        return F.jaccard_loss(
            gt=gt,
            pr=pr,
            class_weights=self.class_weights
        )


class TverskyLoss(Loss):
    def __init__(self, alpha=0.45, beta=0.55, class_weights=None):
        super().__init__(name='tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1.

    def __call__(self, gt, pr):
        return F.tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            class_weights=self.class_weights
        )


class FocalTverskyLoss(Loss):
    def __init__(self, alpha=0.45, beta=0.55, gamma=2.5, class_weights=None):
        super().__init__(name='focal_tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.

    def __call__(self, gt, pr):
        return F.focal_tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            class_weights=self.class_weights
        )


class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2., class_weights=None):
        super().__init__(name='categorical_focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.

    def __call__(self, gt, pr):
        return F.categorical_focal_loss(
            gt,
            pr,
            alpha=self.alpha,
            gamma=self.gamma,
            class_weights=self.class_weights
        )


class LogCoshDiceLoss(Loss):
    def __init__(self, beta=1, class_weights=None):
        super().__init__(name='log_cosh_dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1

    def __call__(self, gt, pr):
        return F.log_cosh_dice_loss(
            gt=gt,
            pr=pr,
            beta=self.beta,
            class_weights=self.class_weights
        )


class LogCoshTverskyLoss(Loss):
    def __init__(self, alpha=0.3, beta=0.7, class_weights=None):
        super().__init__(name='log_cosh_tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1.

    def __call__(self, gt, pr):
        return F.log_cosh_tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            class_weights=self.class_weights
        )


class LogCoshFocalTverskyLoss(Loss):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3, class_weights=None):
        super().__init__(name='log_cosh_focal_tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.

    def __call__(self, gt, pr):
        return F.log_cosh_focal_tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            class_weights=self.class_weights
        )


class LogCoshLoss(Loss):
    def __init__(self, base_loss, **kwargs):
        super().__init__(name=f'log_cosh_{base_loss}')
        self.loss = getattr(F, base_loss)
        self.kwargs = kwargs

    def __call__(self, gt, pr):
        x = self.loss(gt, pr, **self.kwargs)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)
