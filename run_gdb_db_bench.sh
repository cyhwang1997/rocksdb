#!/bin/bash
#DBPATH="/home/cyhwang/rocksdb_filter_test/0131/1638"
DBPATH="/home/cyhwang/rocksdb_filter_test/0201/1755"
#gdb --args db_bench --db=$DBPATH --benchmarks="readrandom" --use_existing_db=true --bloom_bits=10 --write_buffer_size=16777216 --target_file_size_base=16777216 --num=200000 --report_bg_io_stats=true
gdb --args db_bench --db=$DBPATH --benchmarks="fillrandom" --bloom_bits=10 --num=2000000 --report_bg_io_stats=true
