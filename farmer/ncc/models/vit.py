from vit_keras import vit, utils

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GaussianNoise


def ViT(model_name, nb_classes, height, width):

    with tf.device("/cpu:0"):
        base_model = vit.vit_l16(
            include_top=False,
            pretrained=True,
            image_size=height,
            activation='sigmoid',
            pretrained_top=False,
        )
        finetune_at = 28

        # 出力付近以外をフリーズ
        for layer in base_model.layers[:finetune_at - 1]:
            layer.trainable = False

        # ノイズの追加
        noise = GaussianNoise(0.01, input_shape=(height, width, 3))

        model = Sequential()
        model.add(noise)
        model.add(base_model)
        model.add(Dense(nb_classes, activation="softmax"))
                #x = base_model.output
                #x = GlobalAveragePooling2D()(x)
                #predictions = Dense(nb_classes, activation='softmax')(x)
                #model = Model(base_model.input, predictions)

    #return model
    return model