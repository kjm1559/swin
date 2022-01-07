import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from source.utils import window_partition, window_reverse, gelu

class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob
    
    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))

class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim, name='projection')
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim, name='pos_embed')

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)

class PatchMerging(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchMerging, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False, name='linear_trans')

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
        )
    
    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size)) + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

class WindowAttentionV2(WindowAttention):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs):
        super(WindowAttentionV2, self).__init__(dim, window_size, num_heads, qkv_bias, dropout_rate, **kwargs)
        self.cpb = keras.Sequential(
            [
                layers.Dense(512),
                layers.Activation('relu'),
                layers.Dropout(dropout_rate),
                layers.Dense(num_heads),
                layers.Dropout(dropout_rate)
            ]
        )
        
    def build(self, input_shape):
        self.tau = self.add_weight(
            shape=(self.num_heads, self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1]),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        self.log_relative_position_index = tf.math.sign(tf.cast(relative_coords, dtype='float32')) * tf.math.log(tf.abs(tf.cast(relative_coords, dtype='float32')) + 1)
        
    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        qk = tf.linalg.matmul(q, k, transpose_b=True)
        q2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(q * q, -1)), 3)
        k2 = tf.expand_dims(tf.sqrt(tf.reduce_sum(k * k, -1)), 3)
        attn = qk/tf.linalg.matmul(q2, k2, transpose_b=True)
        attn /= self.tau + 0.01
        relative_position_bias = self.cpb(self.log_relative_position_index)
        relative_position_bias = tf.reshape(relative_position_bias,
            (-1, self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1])
        )
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size)) + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv        

class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim, 
        num_patch, 
        num_heads, 
        window_size=7, 
        shift_size=0, 
        num_mlp=1024, 
        qkv_bias=True,
        dropout_rate=0.,
        **kwargs,
    ):
        super(SwinTransformer, self).__init__(**kwargs)
        self.dim = dim # number of input dimesions
        self.num_patch = num_patch # number of embeded patches
        self.num_heads = num_heads # number of attention heads
        self.window_size = window_size # size of window
        self.shift_size = shift_size # size of window shift
        self.num_mlp = num_mlp # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim, 
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate)
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100., attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0., attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x

class SwinTransformerV2(SwinTransformer):
    def __init__(
        self,
        dim, 
        num_patch, 
        num_heads, 
        window_size=7, 
        shift_size=0, 
        num_mlp=1024, 
        qkv_bias=True,
        dropout_rate=0.,
        mlp_ratio=4,
        **kwargs,
    ):
        super(SwinTransformerV2, self).__init__(dim, num_patch, num_heads, window_size, shift_size, num_mlp, qkv_bias, dropout_rate, **kwargs)
        self.attn = WindowAttentionV2(
            dim, 
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.mlp = keras.Sequential(
            [
                layers.Dense(int(dim * mlp_ratio)),
                layers.Activation(gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate)
            ]
        )
    
    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = self.norm1(x)
        x = x_skip + x
        x_skip = x
        x = self.mlp(x)
        x = self.drop_path(x)
        x = self.norm2(x)
        x = x_skip + x
        return x
    
def build_swin_model(
    input_shape,
    image_dimension,
    patch_size,
    num_patch_x,
    num_patch_y,
    embed_dim,
    num_heads,
    window_size,
    num_mlp,
    qkv_bias,
    dropout_rate,
    shift_size,
    num_classes
):
    input = layers.Input(input_shape)
    if tf.__version__ == '2.7.0':
        x = layers.RandomCrop(image_dimension, image_dimension)(input)
        x = layers.RandomFlip("horizontal")(x)
    else:
        x = layers.experimental.preprocessing.RandomCrop(image_dimension, image_dimension)(input)
        x = layers.experimental.preprocessing.RandomFlip("horizontal")(x)
    x = PatchExtract(patch_size)(x)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim, name='patch_embedding')(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(input, output)
    return model

def swin_block(x, embed_dim, num_patch_x, num_patch_y, num_heads, window_size, shift_size, num_mlp, qkv_bias, dropout_rate, count=1, v_flag='v2'):
    for i in range(count * 2):
        if i % 2 == 0:
            tmp_shift_size = 0 
        else:
            tmp_shift_size = shift_size
        if v_flag == 'v1':
            x = SwinTransformer(
                dim=embed_dim,
                num_patch=(num_patch_x, num_patch_y),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=tmp_shift_size,
                num_mlp=num_mlp,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
            )(x)
        else:
            x = SwinTransformerV2(
                dim=embed_dim,
                num_patch=(num_patch_x, num_patch_y),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=tmp_shift_size,
                num_mlp=num_mlp,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
            )(x)
    return x
    

def build_swin_T_model(
    input_shape,
    image_dimension,
    patch_size,
    num_patch_x,
    num_patch_y,
    embed_dim,
    num_heads,
    window_size,
    num_mlp,
    qkv_bias,
    dropout_rate,
    shift_size,
    num_classes,
    v_flag = 'v2',
):
    input = layers.Input(input_shape)
    #tf.image.resize(input, (64, 64))
    if tf.__version__ == '2.7.0':
        x = layers.Resizing((image_dimension, image_dimension), "bilinear")(input)
        x = layers.RandomCrop(image_dimension, image_dimension)(x)
        x = layers.RandomFlip("horizontal")(x)
    else:
        x = layers.experimental.preprocessing.Resizing(image_dimension, image_dimension, "bilinear")(input)
#         x = layers.experimental.preprocessing.RandomCrop(image_dimension, image_dimension)(x)
#         x = layers.experimental.preprocessing.RandomFlip("horizontal")(x)
    x = PatchExtract(patch_size)(x)
    print(f'patch extract shape: {x.shape}')
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim, name='patch_embedding')(x)
    print(f'patch embedding : {x.shape}')
    # stage1
    x = swin_block(x, 
        embed_dim,
        num_patch_x, num_patch_y,
        3,
        window_size,
        shift_size,
        num_mlp,
        qkv_bias,
        dropout_rate,
    )

    # stage2
    print(f'stage1 x shape : {x.shape}')
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='patch_merging1')(x)
    print(f'merging x shape : {x.shape}')
    x = swin_block(x, 
        embed_dim * 2,
        num_patch_x//2, num_patch_y//2,
        6,
        window_size,
        shift_size,
        num_mlp,
        qkv_bias,
        dropout_rate,
    )
    
    # stage3
    print(f'stage2 x shape : {x.shape}')
    x = PatchMerging((num_patch_x//2, num_patch_y//2), embed_dim=embed_dim*2, name='patch_merging2')(x)
    print(f'merging x shape : {x.shape}')
    x = swin_block(x, 
        embed_dim * 4,
        num_patch_x//4, num_patch_y//4,
        12,
        window_size,
        shift_size,
        num_mlp,
        qkv_bias,
        dropout_rate,
        3,
    )
    
    # stage4
    print(f'stage3 x shape : {x.shape}')
    x = PatchMerging((num_patch_x//4, num_patch_y//4), embed_dim=embed_dim*4, name='patch_merging3')(x)
    print(f'merging x shape : {x.shape}')
    x = swin_block(x, 
        embed_dim * 8,
        num_patch_x//8, num_patch_y//8,
        24,
        window_size,
        shift_size,
        num_mlp,
        qkv_bias,
        dropout_rate,
    )

    print(f'stage4 x shape : {x.shape}')
    
    x = PatchMerging((num_patch_x//8, num_patch_y//8), embed_dim=embed_dim * 8, name='patch_merging4')(x)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(input, output)
    return model