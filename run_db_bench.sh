#!/bin/bash

./db_bench --db=/home/cyhwang/rocksdb_filter_test  --benchmarks=fillrandom  --bloom_bits=20 --num=10000000
