# Dataset Statistics

## Full dataset

```text
EVENT TYPE COUNTS:
------------------------------
  product_buy         : 1,775,394
  add_to_cart         : 5,674,064
  remove_from_cart    : 1,937,170
  page_visit          : 156,032,014
  search_query        : 10,218,831

UNIQUE ENTITY COUNTS:
------------------------------
  client_id          : 18,688,704
  sku                : 1,260,365
  category           : 6,774
  url                : 12,650,786
  price              : 100

SEQUENCE LENGTH STATISTICS:
------------------------------
  Number of sequences: 18,688,704
  Mean:               9.40
  Median:             2.00
  Std Dev:            50.28
  Min:                1
  Max:                36004
  Quantiles:
       1%:          1
       5%:          1
      10%:          1
      25%:          1
      50%:          2
      75%:          6
      90%:         18
      95%:         34
      99%:        119

TOTAL STATISTICS:
------------------------------
  Total events       : 175,637,473
```

## sequences_full.parquet

```text
EVENT TYPE COUNTS:
------------------------------
  page_visit          : 156,032,014
  search_query        : 10,218,831
  add_to_cart         : 5,674,064
  remove_from_cart    : 1,937,170
  product_buy         : 1,775,394

UNIQUE ENTITY COUNTS:
------------------------------
  client_id          : 18,688,704
  sku                : 4,536
  category           : 6,403
  url                : 26,874
  price              : 100

SEQUENCE LENGTH STATISTICS:
------------------------------
  Number of sequences: 18,688,704
  Mean:               9.40
  Median:             2.00
  Std Dev:            50.28
  Min:                1
  Max:                36004
  Quantiles:
       1%:          1
       5%:          1
      10%:          1
      25%:          1
      50%:          2
      75%:          6
      90%:         18
      95%:         34
      99%:        119

TOTAL STATISTICS:
------------------------------
  Total events       : 175,637,473
```

## sequences_0.parquet

```text
EVENT TYPE COUNTS:
------------------------------
  page_visit          : 18,406,961
  search_query        : 1,219,189
  add_to_cart         : 715,290
  remove_from_cart    : 232,295
  product_buy         : 196,784

UNIQUE ENTITY COUNTS:
------------------------------
  client_id          : 1,000,000
  sku                : 4,536
  category           : 5,647
  url                : 26,830
  price              : 100

SEQUENCE LENGTH STATISTICS:
------------------------------
  Number of sequences: 1,000,000
  Mean:               20.77
  Median:             11.00
  Std Dev:            26.97
  Min:                1
  Max:                128
  Quantiles:
       1%:          1
       5%:          3
      10%:          4
      25%:          6
      50%:         11
      75%:         22
      90%:         50
      95%:         85
      99%:        128

TOTAL STATISTICS:
------------------------------
  Total events       : 20,770,519
```
