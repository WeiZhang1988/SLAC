import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

### Assuming original image is 224*224*3 rgb, patch size is 16*16*3, patch number is 14*14

from patch_embedding import PatchExtractor, PatchEmbedding
from multihead_attention import MultiHeadAttention
from encoder import Encoder
from decoder import Decoder
from model import TransformerModel

x = tf.ones([2,224,224,3])
print('x: ',tf.shape(x))
pe = PatchExtractor((224,224,3),(16,16,3))
y = pe(x)
print('y: ',tf.shape(y))
mha = MultiHeadAttention(8,512,512,128)
z = mha(y,y,y)
print('z: ',tf.shape(z))


image_size = (224,224,3)
patch_size = (16,16,3)
h = 8
d_k = 512
d_v = 512
d_model = 768
d_ff=1024
n=5
rate=0.0

transformer = TransformerModel(image_size, patch_size, image_size, patch_size, h, d_k, d_v, d_model, d_ff, n, rate)

output = transformer(x,x,True)
print('output: ',tf.shape(output))
