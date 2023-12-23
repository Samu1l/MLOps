# MLOps

# OS
Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Model Identifier: MacBookPro16,1
      Processor Name: 6-Core Intel Core i7
      Processor Speed: 2,6 GHz
      Number of Processors: 1
      Total Number of Cores: 6
      L2 Cache (per Core): 256 KB
      L3 Cache: 12 MB
      Hyper-Threading Technology: Enabled
      Memory: 16 GB
      System Firmware Version: 2020.0.1.0.0 (iBridge: 21.16.365.0.0,0)

hw.physicalcpu: 6
hw.logicalcpu: 12
# Descr

В качестве задачи взял самое простое - [wine classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html), разбиваю данные в ноутбуке `notebooks/data_prepare.ipynb` на `train.csv`, `test.csv` и кладу в папку `data/`, а затем начинаю отслеживать их с помощью dvc.

Чтобы запустить `train.py` или `run_server.py`, необходимо указать `mlflow_uri` в `conf/mlflow.yaml`, либо использовать cli синтаксис hydra для переопределения параметров конфига, то есь вот так:
```bash
python train.py mlflow_uri=http://127.0.0.1:8080
```
После запуска `train.py` будет создана `model_uri` в MLflow Models, которая впоследствии будет использована `run_server.py`. То есть при запуске `run_server.py` будет проскорен файл, лежащий в `infer_data_path` в `conf/infer.yaml`, а предсказания будут сложены в `save_predictions_path`.
`infer.py` делает тоже самое, что и `run_server.py`, только использует локально сохраненную модель.


# Triton

## Дерево model_repository
```
model_repository
├── ensemble-onnx
│   ├── 1
│   └── config.pbtxt
├── onnx-classifier
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
└── python-scaler # для поиграться сделал так
    ├── 1
    │   └── model.py
    └── config.pbtxt
```
## Оптимизация triton
При выполнении команды
```bash
perf_analyzer -m onnx-classifier -u localhost:8500 -concurrency-range 2:2 --shape input:1,13
```
Получал warning от perf-analyzer'а, из-за которого все результаты считались для `Concurrency: 1`
```
[WARNING] Perf Analyzer is not able to keep up with the desired load. The results may not be accurate.
```
Погуглив, пришел к выводу, что проблема в слабом железе компа, на котором все это запускал, поэтому dynamic_batching мерить по сути бессмысленно ((

Без каких либо оптимизаций - Concurrency: 1, throughput: 2680.2 infer/sec, latency 372 usec

Только с dynamic_batching: { } - Concurrency: 1, throughput: 2232.66 infer/sec, latency 447 usec

Только с instance_group.count=2 - Concurrency: 1, throughput: 2848.55 infer/sec, latency 350 usec - выбрал эту конфигурацию.
