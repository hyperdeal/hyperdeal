# Performence tests

## Test matrix_free_loops

This performance test the following matrix-free loop configurations:

| Configuration   | FCL | ECL | SM  | Buffering | Overlapping |
| --------------- |:---:|:---:|:---:|:---------:|:-----------:|
| Configuration 1 |  x  |     |     |    (x)    |     (0)     |
| Congiguration 2 |  x  |     |  x  |     x     |     (0)     |
| Configuration 3 |     |  x  |     |    (x)    |     0/1     |
| Congiguration 4 |     |  x  |  x  |     x     |    0/1/2    |
| Configuration 5 |     |  x  |  x  |           |    0/1/2    |

In total, 10 configurations are tested.
