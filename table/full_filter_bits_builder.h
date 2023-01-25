// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
// Copyright (c) 2012 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rocksdb/filter_policy.h"

namespace rocksdb {

typedef struct __attribute__ ((__packed__)) vqf_block {
  uint64_t md[2];
  uint8_t tags[48];
} vqf_block;

typedef struct vqf_metadata {
  uint64_t total_size_in_bytes;
  uint64_t key_remainder_bits;
  uint64_t range;
  uint64_t nblocks;
  uint64_t nslots;
} vqf_metadata;

typedef struct vqf_filter {
  vqf_metadata metadata;
  vqf_block blocks[];
} vqf_filter;


class Slice;

class FullFilterBitsBuilder : public FilterBitsBuilder {
 public:
  explicit FullFilterBitsBuilder(const size_t bits_per_key,
                                 const size_t num_probes);

  ~FullFilterBitsBuilder();

  virtual void AddKey(const Slice& key) override;

  // Create a filter that for hashes [0, n-1], the filter is allocated here
  // When creating filter, it is ensured that
  // total_bits = num_lines * CACHE_LINE_SIZE * 8
  // dst len is >= 5, 1 for num_probes, 4 for num_lines
  // Then total_bits = (len - 5) * 8, and cache_line_size could be calculated
  // +----------------------------------------------------------------+
  // |              filter data with length total_bits/8              |
  // +----------------------------------------------------------------+
  // |                                                                |
  // | ...                                                            |
  // |                                                                |
  // +----------------------------------------------------------------+
  // | ...                | num_probes : 1 byte | num_lines : 4 bytes |
  // +----------------------------------------------------------------+
  virtual Slice Finish(std::unique_ptr<const char[]>* buf) override;

  // Calculate num of entries fit into a space.
  virtual int CalculateNumEntry(const uint32_t space) override;

  // Calculate space for new filter. This is reverse of CalculateNumEntry.
  uint32_t CalculateSpace(const int num_entry, uint32_t* total_bits,
                          uint32_t* num_lines);

 private:
  friend class FullFilterBlockTest_DuplicateEntries_Test;
  friend class CVQFBlockTest_DuplicateEntries_Test;/*[CYDBG] for cvqf_block_test*/
  size_t bits_per_key_;
  size_t num_probes_;
  std::vector<uint32_t> hash_entries_;

  // Get totalbits that optimized for cpu cache line
  uint32_t GetTotalBitsForLocality(uint32_t total_bits);

  // Reserve space for new filter
  char* ReserveSpace(const int num_entry, uint32_t* total_bits,
                     uint32_t* num_lines);

  // Assuming single threaded access to this function.
  void AddHash(uint32_t h, char* data, uint32_t num_lines, uint32_t total_bits);

  // No Copy allowed
  FullFilterBitsBuilder(const FullFilterBitsBuilder&);
  void operator=(const FullFilterBitsBuilder&);
};

/*CVQF*/
class CVQFBitsBuilder : public FilterBitsBuilder {
 public:
  explicit CVQFBitsBuilder(const size_t bits_per_key,
                           const size_t num_probes, const uint64_t nslots);

  ~CVQFBitsBuilder();

  virtual void AddKey(const Slice& key) override;

  virtual Slice Finish(std::unique_ptr<const char[]>* buf) override;
  void PrintBits(__uint128_t num, int numbits);
  void PrintTags(uint8_t *tags, uint32_t size);
  void PrintBlock(uint64_t block_index);
  void PrintFilter();

  vqf_filter* GetFilter();

  vqf_filter *filter;

 private:
  friend class CVQFBlockTest_DuplicateEntries_Test;/*[CYDBG] for cvqf_block_test*/
  /*Need to be removed*/
  size_t bits_per_key_;
  size_t num_probes_;
  std::vector<uint32_t> hash_entries_;
  /*Need to be removed*/
  uint64_t GetBlockFreeSpace(uint64_t *vector);

  CVQFBitsBuilder(const CVQFBitsBuilder&);
  void operator=(const CVQFBitsBuilder&);

  uint64_t total_blocks;
  uint64_t total_size_in_bytes;
  uint64_t nslots_;
  //vqf_filter *filter;

};

/*CVQF*/
}  // namespace rocksdb
