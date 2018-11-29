import argparse

def _prepare_dataset_arguments(parser):
    parser.add_argument('--dataset_name', default='')
    parser.add_argument('--fname', default='')
    parser.add_argument('--origin', default='')
    parser.add_argument('--file_hash', default='')

    return parser

def _train_arguments(parser):
    parser.add_argument('--dataset_name', default='WIKI_dataset')
    parser.add_argument('--log1p_target', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--splits', type=list, default=[0.7, 0.2, 0.1])
    parser.add_argument('--seed', type=int, default=17)

    parser.add_argument('--model_name', default='se_resnext50_32_4')
    parser.add_argument('--input_shape', type=list, default=[160, 160, 3])
    parser.add_argument('--se_type', default='scSE')
    parser.add_argument('--se_integration', default='standard')
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--activation', default='relu')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer_type', default='adam')
    parser.add_argument('--decay', type=float, default=0.00006)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--augmentations', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--alias', default='')

    return parser

def _sof_predict_evaluate_arguments(parser):
    parser.add_argument('--log1p_target', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--weights', default='../../nn_models/best_SE-ResNeXt-50 (32 x 4d).h5')

    parser.add_argument('--model_name', default='se_resnext50_32_4')
    parser.add_argument('--input_shape', type=list, default=[160, 160, 3])
    parser.add_argument('--se_type', default='scSE')
    parser.add_argument('--se_integration', default='standard')
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--activation', default='relu')

    return parser


def get_args(module_name):
    arguments_dict = {
        'prepare_dataset': _prepare_dataset_arguments,
        'train': _train_arguments,
        'sof_predict_evaluate': _sof_predict_evaluate_arguments
    }
    parser = argparse.ArgumentParser(module_name)
    parser = arguments_dict[module_name](parser)

    return parser.parse_args()