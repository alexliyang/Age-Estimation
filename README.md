# Age-Estimation

## Dependencies
- pyunpack
- dlib, cv2, mtcnn
- keras, tensorflow
- scipy, numpy, pandas
- imgaug
- pyvips

## Usage
All .sh scripts are in the `scripts` folder.

To execute the pipeline from the beginning to the end, you need to run the scripts in the following order:
```sh
prepare_datasets.sh
```
```sh
train.sh
```
```sh
sof_predict_evaluate.sh
```

Logs and results with metrics will be in the `logs` folder.
 The trained models will be in the `nn_models` folder.