import tensorflow as tf
import tensorflow_hub as hub


class MyBiTModel(tf.keras.Model):
  """BiT with a new head."""

  def __init__(self, nb_classes):
    super().__init__()

    self.nb_classes = nb_classes
    self.head = tf.keras.layers.Dense(nb_classes, kernel_initializer='zeros')

    # model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
    # module = hub.KerasLayer(model_url)
    module = hub.KerasLayer("/mnt/cloudy_z/src/atanaka/Downloads/bit_m_r152x4_1")
    self.bit_model = module

  def call(self, images):
    # No need to cut head off since we are using feature extractor model
    bit_embedding = self.bit_model(images)
    return self.head(bit_embedding)

def bit_m_r152x4(nb_classes):
    model = MyBiTModel(nb_classes)
    return model

# poetry add tensorflow_hub
# nohup docker exec -t farmer_tanaka2 bash -c "cd $PWD && Godfarmer" > log.out &