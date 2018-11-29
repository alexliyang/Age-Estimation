from keras.models import Model
from keras.layers import Activation, Dropout
from keras.layers import Input
from keras.layers.core import Dense
from keras.layers.core import Lambda
from keras.layers.core import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.layers.merge import add
from keras.layers.merge import multiply
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.pooling import MaxPooling2D


class SENet(object):

    def __init__(self, input_shape, depth=(3, 8, 36, 3), cardinality=64, bottleneck_width=4, se_type='scSE',
                 se_integration='standard', reduction=16, senet154_modifications=True, activation='relu',
                 kernel_initializer='he_normal', name='SENet-154'):

        self.input_shape = input_shape
        self.depth = depth
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.se_integration = se_integration
        self.reduction = reduction
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.name = name

        if se_type == 'cSE':
            self.se_type = self._cSE_block

        elif se_type == 'sSE':
            self.se_type = self._sSE_block

        elif se_type == 'scSE':
            self.se_type = self._scSE_block

        else:
            raise ValueError('Unknown SE block {0}'.format(se_type))

        if se_integration not in [None, 'standard', 'pre', 'post', 'identity']:
            raise ValueError('Unknown SE integration {0}'.format(se_integration))

        if senet154_modifications:
            self.input_3x3 = True
            self.layer1_out_filters = 128
            self.halve_pointwise = True
            self.dropout = 0.2
            self.downsample_kernel_size = 3

        else:
            self.input_3x3 = False
            self.layer1_out_filters = 64
            self.halve_pointwise = False
            self.dropout = None
            self.downsample_kernel_size = 1

        self.params_per_layers = None
        self._calculate_params_per_layers()


    def _calculate_params_per_layers(self):
        params_per_layers = dict()

        gconv_width = self.cardinality * self.bottleneck_width
        params_per_layers['gconv_out_filters'] = [gconv_width * pow(2, i) for i in range(len(self.depth))]

        if self.halve_pointwise:
            params_per_layers['pointwise_out_filters'] = [filters // 2 for filters in params_per_layers['gconv_out_filters']]
        else:
            params_per_layers['pointwise_out_filters'] = params_per_layers['gconv_out_filters']

        params_per_layers['gconv_stride'] = [1, 2, 2, 2]
        params_per_layers['downsample_stride'] = [1, 2, 2, 2]
        params_per_layers['downsample_kernel_size'] = [1] + [self.downsample_kernel_size] * 3

        self.params_per_layers = params_per_layers

        return self.params_per_layers


    def _cSE_block(self, input_tensor):
        feature_maps = input_tensor._keras_shape[3]

        se = GlobalAveragePooling2D()(input_tensor)
        se = Reshape((1, 1, feature_maps))(se)
        se = Dense(
            feature_maps // self.reduction,
            activation='relu',
            kernel_initializer=self.kernel_initializer,
            use_bias=False)(se)
        se = Dense(
            feature_maps,
            activation='sigmoid',
            kernel_initializer=self.kernel_initializer,
            use_bias=False)(se)

        return multiply([input_tensor, se])


    def _sSE_block(self, input_tensor):
        se = Conv2D(
            1, (1, 1),
            strides=(1, 1),
            activation='sigmoid',
            padding='same',
            use_bias=False)(input_tensor)

        return multiply([input_tensor, se])


    def _scSE_block(self, input_tensor):
        channel_se = self._cSE_block(input_tensor)
        spatial_se = self._sSE_block(input_tensor)

        return add([channel_se, spatial_se])


    def _grouped_conv(self, inputs, out_filters, strides, layer_index, block_index):

        if self.cardinality == 1:
            x = Conv2D(
                out_filters, (3, 3),
                strides=strides,
                padding='same',
                use_bias=False,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                name='conv{0}_{1}_3x3'.format(layer_index, block_index))(inputs)

            return x

        grouped_channels = self.params_per_layers['pointwise_out_filters'][layer_index - 2] // self.cardinality
        group_list = list()
        for c in range(self.cardinality):
            x = Lambda(lambda z: z[..., c*grouped_channels: (c + 1)*grouped_channels])(inputs)
            x = Conv2D(
                out_filters // self.cardinality, (3, 3),
                strides=strides,
                padding='same',
                use_bias=False,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                name='conv{0}_{1}_{2}_3x3'.format(layer_index, block_index, c))(x)
            group_list.append(x)

        return concatenate(group_list, name='conv{0}_{1}_3x3'.format(layer_index, block_index))


    def _create_bottleneck_block(self, inputs, pointwise_out_filters, gconv_out_filters, gconv_stride, layer_index,
                                 block_index, downsample_stride=None, downsample_kernel_size=None):
        layer_index += 2
        block_index += 1
        shortcut = inputs
        if self.se_integration == 'identity':
            shortcut = self.se_type(shortcut)

        if self.se_integration == 'pre':
            inputs = self.se_type(inputs)

        x = Conv2D(
            pointwise_out_filters, (1, 1),
            strides=1,
            padding='same',
            use_bias=False,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            name='conv{0}_{1}_1x1_reduce'.format(layer_index, block_index))(inputs)
        x = BatchNormalization(name='conv{0}_{1}_1x1_reduce/bn'.format(layer_index, block_index))(x)

        x = self._grouped_conv(x, gconv_out_filters, gconv_stride, layer_index, block_index)
        x = BatchNormalization(name='conv{0}_{1}_3x3/bn'.format(layer_index, block_index))(x)

        x = Conv2D(
            pointwise_out_filters * 2, (1, 1),
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name='conv{0}_{1}_1x1_increase'.format(layer_index, block_index))(x)
        x = BatchNormalization(name='conv{0}_{1}_1x1_increase/bn'.format(layer_index, block_index))(x)
        if self.se_integration == 'standard':
            x = self.se_type(x)

        if downsample_stride:
            shortcut = Conv2D(
                pointwise_out_filters * 2,
                kernel_size=downsample_kernel_size,
                strides=downsample_stride,
                padding='same',
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                name='conv{0}_{1}_1x1_proj'.format(layer_index, block_index))(shortcut)

        x = add([x, shortcut], name='add{0}_{1}'.format(layer_index, block_index))
        x = Activation(self.activation, name='add{0}_{1}/activation'.format(layer_index, block_index))(x)

        if self.se_integration == 'post':
            x = self.se_type(x)

        return x


    def _make_init_layer(self, inputs):

        if self.input_3x3:
            x = Conv2D(
                64, (3, 3),
                strides=2,
                padding='same',
                use_bias=False,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                name='conv1_1/3x3_s2')(inputs)
            x = BatchNormalization(name='conv1_1/3x3_s2/bn')(x)

            x = Conv2D(
                64, (3, 3),
                strides=1,
                padding='same',
                use_bias=False,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                name='conv1_2/3x3')(x)
            x = BatchNormalization(name='conv1_2/3x3/bn')(x)

            x = Conv2D(
                self.layer1_out_filters, (3, 3),
                strides=1,
                padding='same',
                use_bias=False,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                name='conv1_3/3x3')(x)
            x = BatchNormalization(name='conv1_3/3x3/bn')(x)

        else:
            x = Conv2D(
                self.layer1_out_filters, (7, 7),
                strides=2,
                padding='same',
                use_bias=False,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                name='conv1_1/7x7_s2')(inputs)
            x = BatchNormalization(name='conv1_1/7x7_s2/bn')(x)

        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool1/3x3_s2')(x)

        return x


    def _make_layer(self, inputs, layer_index):

        x = inputs

        for block_index in range(self.depth[layer_index]):

            if block_index == 0:
                x = self._create_bottleneck_block(
                    inputs=x,
                    pointwise_out_filters=self.params_per_layers['pointwise_out_filters'][layer_index],
                    gconv_out_filters=self.params_per_layers['gconv_out_filters'][layer_index],
                    gconv_stride=self.params_per_layers['gconv_stride'][layer_index],
                    layer_index=layer_index,
                    block_index=block_index,
                    downsample_stride=self.params_per_layers['downsample_stride'][layer_index],
                    downsample_kernel_size=self.params_per_layers['downsample_kernel_size'][layer_index],
                )

            else:
                x = self._create_bottleneck_block(
                    inputs=x,
                    pointwise_out_filters=self.params_per_layers['pointwise_out_filters'][layer_index],
                    gconv_out_filters=self.params_per_layers['gconv_out_filters'][layer_index],
                    gconv_stride=1,
                    layer_index=layer_index,
                    block_index=block_index,
                )

        return x


    def build(self):

        inputs = Input(self.input_shape, name='inputs')

        layer1 = self._make_init_layer(inputs)
        layer2 = self._make_layer(layer1, layer_index=0)
        layer3 = self._make_layer(layer2, layer_index=1)
        layer4 = self._make_layer(layer3, layer_index=2)
        layer5 = self._make_layer(layer4, layer_index=3)
        gmp = GlobalMaxPooling2D(name='globalmaxpooling')(layer5)

        # if self.dropout:
        #     gmp = Dropout(self.dropout, name='dropout')(gmp)

        self.model = Model(inputs=inputs, outputs=gmp, name=self.name)

        return self