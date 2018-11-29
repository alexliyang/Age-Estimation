from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from .senet import SENet

def _add_regressor(model_wrapper, activation, kernel_initializer):

    x = Dropout(0.4)(model_wrapper.model.output)
    x = Dense(1024, activation=activation, kernel_initializer=kernel_initializer, name='fc1')(model_wrapper.model.output)
    x = BatchNormalization(name='fc1/bn')(x)
    x = Dense(512, activation=activation, kernel_initializer=kernel_initializer, name='fc2')(x)
    x = BatchNormalization(name='fc2/bn')(x)
    x = Dense(128, activation=activation, kernel_initializer=kernel_initializer, name='fc3')(x)
    x = BatchNormalization(name='fc3/bn')(x)
    x = Dense(1, kernel_initializer=kernel_initializer, name='output')(x)

    model_wrapper.model = Model(model_wrapper.model.input, x)
    return model_wrapper


def make_model(name, input_shape=(224, 224, 3), se_type='scSE', se_integration='standard', reduction=16,
               activation='relu', kernel_initializer='he_normal'):

    if name == 'se_resnext50_32_4':
        model_wrapper = SENet(
            input_shape=input_shape,
            depth=[3, 4, 6, 3],
            cardinality=32,
            bottleneck_width=4,
            se_type=se_type,
            se_integration=se_integration,
            reduction=reduction,
            senet154_modifications=False,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name='SE-ResNeXt-50 (32 x 4d)'
        )
        model_wrapper.build()
        return _add_regressor(model_wrapper, activation, kernel_initializer)


    elif name == 'se_resnext101_32_4':
        model_wrapper = SENet(
            input_shape=input_shape,
            depth=[3, 4, 23, 3],
            cardinality=32,
            bottleneck_width=4,
            se_type=se_type,
            se_integration=se_integration,
            reduction=reduction,
            senet154_modifications=False,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name='SE-ResNeXt-101 (32 x 4d)'
        )
        model_wrapper.build()
        return _add_regressor(model_wrapper, activation, kernel_initializer)

    elif name == 'se_resnext152_32_4':
        model_wrapper = SENet(
            input_shape=input_shape,
            depth=[3, 8, 36, 3],
            cardinality=32,
            bottleneck_width=4,
            se_type=se_type,
            se_integration=se_integration,
            reduction=reduction,
            senet154_modifications=False,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name='SE-ResNeXt-152 (32 x 4d)'
        )
        model_wrapper.build()
        return _add_regressor(model_wrapper, activation, kernel_initializer)

    elif name == 'senet154':
        model_wrapper = SENet(
            input_shape=input_shape,
            depth=[3, 8, 36, 3],
            cardinality=64,
            bottleneck_width=4,
            se_type=se_type,
            se_integration=se_integration,
            reduction=reduction,
            senet154_modifications=True,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name='SENet-154'
        )
        model_wrapper.build()
        return _add_regressor(model_wrapper, activation, kernel_initializer)

    else:
        raise ValueError('Unknown network "{0}"'.format(name))