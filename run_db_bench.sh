#!/bin/bash

#./db_bench --db=/home/cyhwang/rocksdb_filter_test/test  --benchmarks="fillrandom,stats" --num=1000000 --threads=4 --statistics=true --report_bg_io_stats=true > /home/cyhwang/rocksdb_filter_test/test/fillrandom_report.csv
#./db_bench --db=/home/cyhwang/rocksdb_filter_test/20b_100M_fillrandom --benchmarks="readrandom,stats" --use_existing_db=true --bloom_bits=20 --num=10000000 --threads=4 --statistics=true --report_bg_io_stats=true > /home/cyhwang/rocksdb_filter_test/20b_100M_fillrandom/readrandom_1118_report.csv
./db_bench --db=/home/cyhwang/rocksdb_filter_test/  --benchmarks="fillrandom,stats" --num=1000000 --threads=4 --statistics=true --report_bg_io_stats=true > /home/cyhwang/rocksdb_filter_test/test/fillrandom_report.csv
#./db_bench --db=/home/cyhwang/rocksdb_filter_test/20b_100M_fillrandom --benchmarks="readrandom,stats" --use_existing_db=true --bloom_bits=20 --num=10000000 --threads=4 --statistics=true --report_bg_io_stats=true > /home/cyhwang/rocksdb_filter_test/20b_100M_fillrandom/readrandom_1118_report.csv
