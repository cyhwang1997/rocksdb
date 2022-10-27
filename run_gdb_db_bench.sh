#!/bin/bash

gdb --args db_bench --benchmarks="fillrandom" -bloom_bits=20
