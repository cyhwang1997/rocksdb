#!/bin/bash

#./db_bench --db=/home/cyhwang/rocksdb_filter_test/dbbenchtest/option3  --benchmarks="fillrandom,stats" --num=10000000 --threads=4 --statistics=true --stats_interval_seconds=10 --report_bg_io_stats=false --report_file_operations=true 1> /home/cyhwang/rocksdb_filter_test/dbbenchtest/option3/0b_fr_report 2>/home/cyhwang/rocksdb_filter_test/dbbenchtest/option3/0b_fr_Throughput
#./db_bench --db=/home/cyhwang/rocksdb_filter_test/20b_100M_fillrandom --benchmarks="readrandom,stats" --use_existing_db=true --bloom_bits=20 --num=10000000 --threads=4 --statistics=true --report_bg_io_stats=true > /home/cyhwang/rocksdb_filter_test/20b_100M_fillrandom/readrandom_1118_report.csv
./db_bench --db=/home/cyhwang/rocksdb_filter_test/0323 --benchmarks="fillrandom,stats" --bloom_bits=10 --num=2000000 --statistics=true --stats_interval_seconds=10 --report_bg_io_stats=true --report_file_operations=true 1> /home/cyhwang/rocksdb_filter_test/0323/fr_report 2>/home/cyhwang/rocksdb_filter_test/0323/fr_report2
#./db_bench --db=/home/cyhwang/rocksdb_filter_test/0131/1954  --benchmarks="readrandom,stats" --use_existing_db=true --bloom_bits=10 --write_buffer_size=16777216 --target_file_size_base=16777216 --num=100000 --statistics=true --stats_interval_seconds=10 --report_bg_io_stats=true --report_file_operations=true 1> /home/cyhwang/rocksdb_filter_test/0131/1954/rr_report1 2>/home/cyhwang/rocksdb_filter_test/0131/1954/rr_report2
