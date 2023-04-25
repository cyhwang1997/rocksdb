#!/bin/bash

#sudo umount /mnt
#sudo mkfs.ext4 /dev/nvme1n1
#sudo mount /dev/nvme1n1 /mnt
#sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"

./db_bench --db=/mnt --benchmarks="fillrandom,stats" --bloom_bits=20 --num=10000000 --key_size=23 --value_size=1000 --write_buffer_size=134217728 --max_background_jobs=20 --max_write_buffer_number=8 --threads=8 --statistics=true --stats_interval_seconds=10 --report_bg_io_stats=true --report_file_operations=true 1> /mnt/fr_report 2>/mnt/fr_report2
sudo sh -c "/usr/bin/echo 3 > /proc/sys/vm/drop_caches"
./db_bench --db=/mnt --benchmarks="readrandom,stats" --use_existing_db=true --bloom_bits=20 --num=10000000 --key_size=23 --value_size=1000 --write_buffer_size=134217728 --max_background_jobs=20 --max_write_buffer_number=8 --threads=8 --statistics=true --stats_interval_seconds=10 --report_bg_io_stats=true --report_file_operations=true 1> /mnt/rr_report 2>/mnt/rr_report2
