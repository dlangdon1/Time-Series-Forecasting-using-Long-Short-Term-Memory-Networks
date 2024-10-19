# Recurrent Neural Network for Time Series Forecasting

<a href="url" align="center"><img src="https://github.com/dlangdon1/Time-Series-Forecasting-using-Long-Short-Term-Memory-Networks/blob/main/utils/banner_00.png" align="center" height="304" width="800" ></a>

---

This is a time series forecasting project based on the Wikipedia [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) dataset from Kaggle. 
Two **RNN architectures** are implemented:
- A "Vanilla" RNN regressor.
- A Seq2seq regressor.

Both are implemented in **TensorFlow 2**, with *custom training functions* optimized with **Autograph**.

## Structure of the repository
Main files:
- `config.yaml`: config file for hyperparameters.
- `dataprep.py`: data preprocessing pipeline.
- `train.py`: training pipeline.
- `tools.py`: contains useful processing functions to be iterated in main pipelines.
- `model.py`: builds model.

I also added a `visualize_performance.ipynb` Jupyter Notebook to visually inspect models' performance on Test data.

Folders:
- `/data_raw/`: requires unzipped `train_2.csv` file from [Kaggle](https://www.kaggle.com/c/web-traffic-time-series-forecasting/). Available is an `imputed.csv` dataset, containing imputed time series, coming from my other repository on a [GAN for imputation of missing data in time series](https://github.com/dlangdon1/Convolutional-Recurrent-Seq2seq-GAN-for-the-Imputation-in-Time-Series-Data.git).
- `/data_processed/`: divided in `/Train/` and `/Test/` directories.
- `/saved_models/`: contains all saved TensorFlow models, both regressors.
- `/utils/`: for pics and other secondary files.

## How to run code
After you clone the repository locally, download the raw dataset from [Kaggle](https://www.kaggle.com/c/web-traffic-time-series-forecasting/), and place unzipped `train_2.csv` file in `/data_raw/` folder.
Then, time series forecast is executed in two steps. First, run data preprocessing pipeline:

`python -m dataprep`

This will generate Training+Validation and Test files, stored in `/data_processed/` subdirectories. Second, launch training pipeline with:

`python -m train`

This will either create, train and save a new model, or load and train an already existing one, stored in `/saved_models/` folder.

Finally, Test set performance will be evaluated from `test.ipynb` notebook.


## Modules
```
numpy==2.1.2
pandas==2.2.3
scikit-learn==1.5.1.post1
scipy==1.14.1
tensorflow==2.16.1
tqdm==4.66.5
```

## Hardware
I used a pretty powerful laptop, with 32GB or RAM and NVidia RTX 3070 GPU. I highly recommend GPU training to avoid excessive computational times.
