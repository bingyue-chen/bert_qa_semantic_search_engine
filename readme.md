# Try Bert - QA semantic search engine

## ref
- [bert-as-service](https://github.com/hanxiao/bert-as-service)


## usage

1. first start bert server
```
bert-serving-start -model_dir models/{downloaded_model_dir}/ -num_worker=2
```

2. after start bert server
```
python search.py
```