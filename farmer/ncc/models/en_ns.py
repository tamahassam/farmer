from efficientnet.tfkeras import EfficientNetB7
# from efficientnet.keras import EfficientNetB7
# from efficientnet.tfkeras import center_crop_and_resize, preprocess_input

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def en_ns(model_name, nb_classes, height, width):
    # model_name_converted = model_name.replace('efficientnetb', 'EfficientNetB')

    with tf.device("/cpu:0"):
        # base_model = getattr(en, model_name_converted)(
        base_model = EfficientNetB7(
            include_top=False,
            input_shape=(height, width, 3),
            weights='noisy-student'
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    return model
