from data import Dataset
from models import make_model
from processor import Trainer
from utils.params import get_args
from keras.layers.advanced_activations import LeakyReLU


if __name__ == '__main__':

    args = get_args('train')

    print('[INFO] Preparing Data...')
    dataset = Dataset(args.dataset_name, log1p_target=args.log1p_target)
    dataset.split_train_test_valid(args.splits, seed=args.seed)

    print('[INFO] Building Model...')
    model_wrapper = make_model(
        args.model_name,
        input_shape=args.input_shape,
        se_type=args.se_type,
        se_integration=args.se_integration,
        reduction=args.reduction,
        activation=LeakyReLU(),
    )

    print('[INFO] Training...')
    trainer = Trainer(
        model_wrapper,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer_type,
        learning_rate=args.learning_rate,
        decay=args.decay,
        augmentations=args.augmentations,
        early_stopping_patience=args.early_stopping_patience,
        alias=args.alias
    )
    trainer.train()
    print('[INFO] Finishing...')