import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import PIL
import PIL.Image
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input, Layer, LayerNormalization, MultiHeadAttention

class PatchExtractor(Layer):
    def __init__(self):
        super(PatchExtractor, self).__init__()

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, 16, 16, 1],
            strides=[1, 16, 16, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
        
class PatchEncoder(Layer):
    def __init__(self, num_patches=196, projection_dim=768):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, \
        trainable=True)
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches+1, \
        output_dim=projection_dim)
        #input_dim means Size of the vocabulary, not really the input

    def call(self, patches):
        batch = tf.shape(patches)[0]
        # reshape the class token embedins
        class_token = tf.tile(self.class_token, multiples = [batch, 1])
        class_token = tf.reshape(class_token, (batch, 1, \
        self.projection_dim))
        # calculate patches embeddings
        patches_embed = self.projection(patches)
        patches_embed = tf.concat([patches_embed, class_token], 1)
        # calcualte positional embeddings
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
        positions_embed = self.position_embedding(positions)
        # add both embeddings
        encoded = patches_embed + positions_embed
        return encoded        
 
class MLP(Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.dense1 = Dense(hidden_features, activation=tf.nn.gelu)
        self.dense2 = Dense(out_features)
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y

class Block(Layer):
    def __init__(self, projection_dim, num_heads=4, dropout_rate=0.1):
        super(Block, self).__init__()
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(num_heads=num_heads, \
        key_dim=projection_dim, dropout=dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)

    def call(self, x):
        # Layer normalization 1.
        x1 = self.norm1(x) # encoded_patches
        # Create a multi-head attention layer.
        attention_output = self.attn(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, x]) #encoded_patches
        # Layer normalization 2.
        x3 = self.norm2(x2)
        # MLP.
        x3 = self.mlp(x3)
        # Skip connection 2.
        y = Add()([x3, x2])
        return y        

class TransformerEncoder(Layer):
    def __init__(self, projection_dim, num_heads=4, num_blocks=12, \
    dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.blocks = [Block(projection_dim, num_heads, dropout_rate) \
        for _ in range(num_blocks)]
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.5)

    def call(self, x):
        # Create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        y = self.dropout(x)
        return y

def create_VisionTransformer(num_classes, num_patches=196, projection_dim=768, input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    # Patch extractor
    patches = PatchExtractor()(inputs)
    # Patch encoder
    patches_embed = PatchEncoder(num_patches, projection_dim)(patches)
    # Transformer encoder
    representation = TransformerEncoder(projection_dim)(patches_embed)
    representation = GlobalAveragePooling1D()(representation)
    # MLP to classify outputs
    logits = MLP(projection_dim, num_classes, 0.5)(representation)
    # Create model
    model = Model(inputs=inputs, outputs=logits)
    return model

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def validate_above_code():
    image = plt.imread('flower.jpeg')
    image = tf.image.resize(tf.convert_to_tensor(image), size=(224, 224))
    plt.imshow(image.numpy().astype("uint8"))
    plt.axis("off");

    batch = tf.expand_dims(image, axis=0)
    patches = PatchExtractor()(batch)
    print(patches.shape)

    n = int(np.sqrt(patches.shape[1]))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (16, 16, 3))
        ax.imshow(patch_img.numpy().astype("uint8"))
        ax.axis("off")
    plt.show()

    embeddings = PatchEncoder()(patches)
    print(embeddings.shape)

    mlp = MLP(768 * 2, 768)
    y = mlp(tf.zeros((1, 197, 768)))
    print(y.shape)

    block = Block(768)
    y = block(tf.zeros((1, 197, 768)))
    print(y.shape)

    transformer = TransformerEncoder(768)
    y = transformer(embeddings)
    print(y.shape)

    model = create_VisionTransformer(2)
    model.summary()
    
def prepare_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url, \
    fname='flower_photos', untar=True)
    data_dir = pathlib.Path(data_dir)
    
def exam_downloaded_dataset(file_dir):
    data_dir = pathlib.Path(file_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    roses = list(data_dir.glob('roses/*'))
    img = PIL.Image.open(str(roses[0]))
    img.show()
    del img
    roses = list(data_dir.glob('roses/*'))
    img = PIL.Image.open(str(roses[1]))
    img.show()
    del img
    
def load_dataset(file_dir,batch_size,img_height,img_width):
    data_dir = pathlib.Path(file_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, \
    validation_split=0.2,subset="training",seed=123, \
    image_size=(img_height, img_width),batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, \
    validation_split=0.2,subset="validation",seed=123, \
    image_size=(img_height, img_width),batch_size=batch_size)
    return train_ds, val_ds

def exam_loaded_dataset(train_ds, val_ds):
    class_names = train_ds.class_names
    print(class_names)
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()
    
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

if __name__ == "__main__":
    batch = 32
    img_height = 224
    img_width = 224
    num_class = 5
    model = create_VisionTransformer(num_class)
    train_ds, val_ds = load_dataset('flower_photos',\
    batch,img_height,img_width)
    model.compile(optimizer='adam',\
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\
    metrics=['accuracy'])
    model.fit(train_ds,validation_data=val_ds,epochs=3)
