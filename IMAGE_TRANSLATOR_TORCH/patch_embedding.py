import tensorflow as tf
from tensorflow import convert_to_tensor, string
from tensorflow.keras.layers import Layer, Dense, Embedding, Reshape
from tensorflow.data import Dataset
import numpy as np

class PatchExtractor(Layer):
    def __init__(self, image_size, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_num_row = image_size[0]//patch_size[0]
        self.patch_num_col = image_size[1]//patch_size[1]
        self.patch_num = self.patch_num_row * self.patch_num_col
        assert image_size[0] == (image_size[0]//patch_size[0]) * patch_size[0]
        assert image_size[1] == (image_size[1]//patch_size[1]) * patch_size[1]

        # Assuming the image has three channels each patch would be
        # of size (patch_size[0], patch_size[1], 3).
        self.resize = Reshape((-1, patch_size[0] * patch_size[1] * patch_size[2]))

    def call(self, images):
        assert np.shape(images)[-3:] == self.image_size 
        # Create patches from the input images
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Reshape the patches to (batch, num_patches, patch_area) and return it.
        patches = self.resize(patches)
        return patches
        
    def reconstruct_from_patch_single_img(self, patch):
        patch = tf.reshape(patch, (self.patch_num, self.patch_size[0], self.patch_size[1], self.patch_size[2]))
        rows = tf.split(patch, self.patch_num_row, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed
        
    def reconstruct_from_patches(self, patches):
        reconstructed = []
        for patch in patches:
            img = self.reconstruct_from_patch_single_img(patch)
            reconstructed.append(img)
        return tf.convert_to_tensor(reconstructed)
        
class PatchEmbedding(Layer):
    def __init__(self, image_size, patch_size, output_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_num = (image_size[0]//patch_size[0]) * (image_size[1]//patch_size[1])
        assert image_size[0] == (image_size[0]//patch_size[0]) * patch_size[0]
        assert image_size[1] == (image_size[1]//patch_size[1]) * patch_size[1]
        self.patch_extractor = PatchExtractor(image_size, patch_size)

        self.word_embedding_layer = Dense(output_dim)
        
        position_embedding_matrix = self.get_position_encoding(self.patch_num, output_dim)
        self.position_embedding_layer = Embedding(
            input_dim=self.patch_num, output_dim=output_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )
        
    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
    
    def call(self, inputs):   
        extracted_inputs = self.patch_extractor(inputs)
        position_indices = tf.range(self.patch_num)
        embedded_words = self.word_embedding_layer(extracted_inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices
