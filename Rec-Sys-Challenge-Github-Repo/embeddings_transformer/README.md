## Run

```shell
python -m embeddings_transformer.data_processor --data-dir ../data/original/ --output-dir ../data/sequence/
python -m embeddings_transformer.model_training --data-dir ../data/original/ --output-dir ../models/ --sequences-path ../data/sequence/

python -m embeddings_transformer.create_embeddings --data-dir ../data/original/ --embeddings-dir ../results/transformer/v2/ --sequences-path ../data/sequence/ --checkpoint-path lightning_logs/version_3/checkpoints/epoch=29-step=210960.ckpt
python -m validator.run --data-dir ../data/original/ --embeddings-dir ../results/transformer/v1/

python -m training_pipeline.train --data-dir ../data/original/ --embeddings-dir ../results/transformer/v2/ --tasks churn propensity_category propensity_sku --log-name baseline --accelerator gpu --devices 0 --disable-relevant-clients-check
```

## Dataset Statistics

number of unique urls: 10212493  
number of unique product categories: 6622  
product name embedding length: 16, embeddings: [175  21 154 157  93 239  71  76 193  95 235  91   8 174 235 154]  
search query embedding length: 16, embeddings: [202 151 206 232 232 181 181  55   9  57 236  97  94 169 198 236]

Number of sequences: 1000000

Sequence Lengths:
  Mean:     51.54
  Median:   15.00
  Std Dev:  166.58
  Min:      1
  Max:      36004
  Quantiles:
0.01      1.0
0.05      1.0
0.10      1.0
0.25      3.0
0.50     15.0
0.75     47.0
0.90    117.0
0.95    201.0
0.99    560.0
dtype: float64

Time Deltas:
  Mean:     53181.96
  Median:   10.00
  Std Dev:  376472.06
  Min:      0.0
  Max:      12068170.0
  Quantiles:
0.01          0.0
0.05          1.0
0.10          1.0
0.25          1.0
0.50         10.0
0.75         60.0
0.90       3030.0
0.95     121855.0
0.99    1477835.0
Name: time_delta, dtype: float64

Event Type (Top 10):
event_type
7    41790902
8     4460870
5     2770295
4     1304215
6     1213319
Name: count, dtype: int64

Event Type (Bottom 5):
event_type
7    41790902
8     4460870
5     2770295
4     1304215
6     1213319
Name: count, dtype: int64
  Unique event types: 5
  Most frequent: 7 (41790902)
  Least frequent: 6 (1213319)

Price (Top 10):
price
4     68034
5     66522
11    64110
22    63847
46    63780
10    62690
8     62193
15    60855
18    60526
13    60398
Name: count, dtype: int64

Price (Bottom 5):
price
85    46147
79    46115
99    45589
62    43659
45    43507
Name: count, dtype: int64
  Unique prices: 100
  Most frequent: 4 (68034)
  Least frequent: 45 (43507)

Category (Top 10):
category
5     127221
4     112535
6      82589
7      61191
8      60519
9      56997
10     53907
12     35064
15     34149
17     33071
Name: count, dtype: int64

Category (Bottom 5):
category
6315    1
6021    1
6144    1
5285    1
5855    1
Name: count, dtype: int64
  Unique categorys: 6274
  Most frequent: 5 (127221)
  Least frequent: 6220 (1)

SKU (Top 10):
sku
1     4080486
4        6831
5        5641
6        5481
7        4826
9        3776
8        3704
12       2844
11       2709
14       2665
Name: count, dtype: int64

SKU (Bottom 5):
sku
3772    64
4535    64
2683    63
4356    51
4355    50
Name: count, dtype: int64
  Unique skus: 4537
  Most frequent: 1 (4080486)
  Least frequent: 4355 (50)

URL (Top 10):
url
1     18924923
4      5245061
5      2422088
7      1370540
6      1181704
9       973193
8       744537
11      621001
10      449437
13      215009
Name: count, dtype: int64

URL (Bottom 10):
url
20601    7
23457    7
25411    6
25326    6
19037    5
24569    5
15242    4
14854    4
23394    3
12503    1
Name: count, dtype: int64
  Unique urls: 26855
  Most frequent: 1 (18924923)
  Least frequent: 12503 (1)
