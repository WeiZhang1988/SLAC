from encoder import Encoder
from decoder import Decoder
import tensorflow as tf
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer, Reshape

class TransformerModel(Model):
    def __init__(self, enc_image_size, enc_patch_size, dec_image_size, dec_patch_size, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)

        # Set up the encoder
        self.encoder = Encoder(enc_image_size, enc_patch_size, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Set up the decoder
        self.decoder = Decoder(dec_image_size, dec_patch_size, h, d_k, d_v, d_model, d_ff_inner, n, rate)

    def padding_mask(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]

    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, encoder_input, decoder_input, training):

        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
        enc_padding_mask = None#self.padding_mask(encoder_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = None#self.padding_mask(decoder_input)
        dec_in_lookahead_mask = None#self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = None#maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)

        model_output = self.decoder.pos_encoding.patch_extractor.reconstruct_from_patches(decoder_output)
        return model_output
