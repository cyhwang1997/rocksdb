#!/bin/bash

gdb --args db_bench --db=/home/cyhwang/rocksdb_filter_test/0131/1638 --benchmarks="readrandom" --use_existing_db=true --bloom_bits=10 --write_buffer_size=16777216 --target_file_size_base=16777216 --num=200000 --report_bg_io_stats=true
#gdb --args db_bench --db=/home/cyhwang/rocksdb_filter_test/0131 --benchmarks="fillrandom" --bloom_bits=10 --write_buffer_size=16777216 --target_file_size_base=16777216 --num=800000 --report_bg_io_stats=true
