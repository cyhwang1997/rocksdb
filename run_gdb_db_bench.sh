#!/bin/bash

gdb --args db_bench --benchmarks="fillrandom" --bloom_bits=0 --num=5000000 --report_bg_io_stats=true
