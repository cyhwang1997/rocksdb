sudo make jclean clean
sudo make rocksdbjava -j12
cp ./java/target/rocksdbjni*.jar ../YCSB/rocksdb_binding/rocksdbjni-cvqf.jar
