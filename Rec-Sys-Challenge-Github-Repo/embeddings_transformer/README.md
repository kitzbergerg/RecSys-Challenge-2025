## Statistics

number of unique urls: 10212493  
number of unique product categories: 6622  
product name embedding length: 16, embeddings: [175  21 154 157  93 239  71  76 193  95 235  91   8 174 235 154]  
search query embedding length: 16, embeddings: [202 151 206 232 232 181 181  55   9  57 236  97  94 169 198 236]

Sequence Length Stats:

- Mean:     51.54
- Median:   15.00
- Std Dev:  166.58
- Max:      36004
- Min:      1
- Quantiles:
    - 0.25 3.0
    - 0.50 15.0
    - 0.75 47.0
    - 0.90 117.0
    - 0.95 201.0
    - 0.99 560.0

## Run on server

```shell
DATA_DIR=../../../../shared/194.035-2025S/data/group_project/data_new/
python -m embeddings_transformer.data_processor --data-dir $DATA_DIR --output-dir ../data/sequence/ --rebuild-vocab
python -m embeddings_transformer.model --data-dir $DATA_DIR --output-dir ../models/ --vocab-file ../data/sequence/vocabularies.pkl --sequences-file ../data/sequence/sequences.pkl
```
