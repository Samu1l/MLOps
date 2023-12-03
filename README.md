# MLOps

В качестве задачи взял самое простое - [wine classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html), разбиваю данные в ноутбуке `notebooks/data_prepare.ipynb` на `train.csv`, `test.csv` и кладу в папку `data/`, а затем начинаю отслеживать их с помощью dvc.

Чтобы запустить `train.py` или `run_server.py`, необходимо указать `mlflow_uri` в `conf/mlflow.yaml`, либо использовать cli синтаксис hydra для переопределения параметров конфига, то есь вот так:
```bash
python train.py mlflow_uri=http://127.0.0.1:8080
```
После запуска `train.py` будет создана `model_uri` в MLflow Models, которая впоследствии будет использована `run_server.py`. То есть при запуске `run_server.py` будет проскорен файл, лежащий в `infer_data_path` в `conf/infer.yaml`, а предсказания будут сложены в `save_predictions_path`.
`infer.py` делает тоже самое, что и `run_server.py`, только использует локально сохраненную модель.
