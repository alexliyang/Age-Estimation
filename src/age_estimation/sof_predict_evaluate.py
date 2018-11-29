import numpy as np
from data import Dataset
from models import make_model
from processor import Inferer
from sklearn.metrics import mean_absolute_error
from keras.layers.advanced_activations import LeakyReLU
from utils import get_args


if __name__ == '__main__':
    args = get_args('sof_predict_evaluate')

    print('[INFO] Preparing Data...')
    sof_dataset = Dataset('SoF_dataset', log1p_target=args.log1p_target)

    print('[INFO] Building Model...')
    model_wrapper = make_model(
        args.model_name,
        input_shape=args.input_shape,
        se_type=args.se_type,
        se_integration=args.se_integration,
        reduction=args.reduction,
        activation=LeakyReLU(),
    )
    model_wrapper.model.load_weights(args.weights)

    print('[INFO] Predicting...')
    inferer = Inferer(model_wrapper, dataset=sof_dataset)
    predictions_dict = inferer.predict()

    labels = list()
    predictions = list()

    for image_name in list(predictions_dict.keys()):
        label = round(np.expm1(sof_dataset.labels_dict[image_name]))\
            if args.log1p_target\
            else sof_dataset.labels_dict[image_name]

        prediction = round(np.expm1(predictions_dict[image_name]))\
            if args.log1p_target\
            else predictions_dict[image_name]

        labels.append(label)
        predictions.append(prediction)

    print('[INFO] Evaluating...')
    mae = mean_absolute_error(labels, predictions)
    print('MAE score on SoF dataset: {0}'.format(mae))