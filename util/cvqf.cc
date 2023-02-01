//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2012 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "rocksdb/filter_policy.h"

#include "rocksdb/slice.h"
#include "table/block_based_filter_block.h"
#include "table/full_filter_bits_builder.h"
#include "table/full_filter_block.h"
#include "util/coding.h"
#include "util/hash.h"

#include <immintrin.h>

namespace rocksdb {

class BlockBasedFilterBlockBuilder;
class FullFilterBlockBuilder;

#define TAG_MASK 0xff
#define QUQU_SLOTS_PER_BLOCK 48
#define QUQU_BUCKETS_PER_BLOCK 80
#define QUQU_CHECK_ALT 92
#define LOCK_MASK (1ULL << 63)
#define QUQU_MAX 255
#define QUQU_PRESLOT 16

/*static inline void lock(vqf_block& block) {
  uint64_t *data;
  data = block.md + 1;
  while ((__sync_fetch_and_or(data, LOCK_MASK) & (1ULL << 63)) != 0) {}
}*/

static uint64_t pre_one[64 + 128] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   1ULL << 0, 1ULL << 1, 1ULL << 2, 1ULL << 3, 1ULL << 4, 1ULL << 5, 1ULL << 6, 1ULL << 7, 1ULL << 8, 1ULL << 9,
   1ULL << 10, 1ULL << 11, 1ULL << 12, 1ULL << 13, 1ULL << 14, 1ULL << 15, 1ULL << 16, 1ULL << 17, 1ULL << 18, 1ULL << 19, 
   1ULL << 20, 1ULL << 21, 1ULL << 22, 1ULL << 23, 1ULL << 24, 1ULL << 25, 1ULL << 26, 1ULL << 27, 1ULL << 28, 1ULL << 29, 
   1ULL << 30, 1ULL << 31, 1ULL << 32, 1ULL << 33, 1ULL << 34, 1ULL << 35, 1ULL << 36, 1ULL << 37, 1ULL << 38, 1ULL << 39, 
   1ULL << 40, 1ULL << 41, 1ULL << 42, 1ULL << 43, 1ULL << 44, 1ULL << 45, 1ULL << 46, 1ULL << 47, 1ULL << 48, 1ULL << 49, 
   1ULL << 50, 1ULL << 51, 1ULL << 52, 1ULL << 53, 1ULL << 54, 1ULL << 55, 1ULL << 56, 1ULL << 57, 1ULL << 58, 1ULL << 59, 
   1ULL << 60, 1ULL << 61, 1ULL << 62, 1ULL << 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

static uint64_t *one = pre_one + 64;

const static uint64_t carry_pdep_table[128] {
    1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL,
    1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL,
    1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 
    1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL,
    1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL,
    1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL,
    1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL,
    1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL, 1ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL
};

const static uint64_t high_order_pdep_table[128] {
    ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0),
    ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0),
    ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0),
    ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0),
    ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0),
    ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0),
    ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0),
    ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0), ~(1ULL << 0),
    ~(1ULL << 0), ~(1ULL << 1), ~(1ULL << 2), ~(1ULL << 3), ~(1ULL << 4), ~(1ULL << 5), ~(1ULL << 6), ~(1ULL << 7),
    ~(1ULL << 8), ~(1ULL << 9), ~(1ULL << 10), ~(1ULL << 11), ~(1ULL << 12), ~(1ULL << 13), ~(1ULL << 14), ~(1ULL << 15),
    ~(1ULL << 16), ~(1ULL << 17), ~(1ULL << 18), ~(1ULL << 19), ~(1ULL << 20), ~(1ULL << 21), ~(1ULL << 22), ~(1ULL << 23),
    ~(1ULL << 24), ~(1ULL << 25), ~(1ULL << 26), ~(1ULL << 27), ~(1ULL << 28), ~(1ULL << 29), ~(1ULL << 30), ~(1ULL << 31),
    ~(1ULL << 32), ~(1ULL << 33), ~(1ULL << 34), ~(1ULL << 35), ~(1ULL << 36), ~(1ULL << 37), ~(1ULL << 38), ~(1ULL << 39),
    ~(1ULL << 40), ~(1ULL << 41), ~(1ULL << 42), ~(1ULL << 43), ~(1ULL << 44), ~(1ULL << 45), ~(1ULL << 46), ~(1ULL << 47),
    ~(1ULL << 48), ~(1ULL << 49), ~(1ULL << 50), ~(1ULL << 51), ~(1ULL << 52), ~(1ULL << 53), ~(1ULL << 54), ~(1ULL << 55),
    ~(1ULL << 56), ~(1ULL << 57), ~(1ULL << 58), ~(1ULL << 59), ~(1ULL << 60), ~(1ULL << 61), ~(1ULL << 62), ~(1ULL << 63)
};

const static uint64_t low_order_pdep_table[128] {
    ~(1ULL << 0), ~(1ULL << 1), ~(1ULL << 2), ~(1ULL << 3), ~(1ULL << 4), ~(1ULL << 5), ~(1ULL << 6), ~(1ULL << 7),
    ~(1ULL << 8), ~(1ULL << 9), ~(1ULL << 10), ~(1ULL << 11), ~(1ULL << 12), ~(1ULL << 13), ~(1ULL << 14), ~(1ULL << 15),
    ~(1ULL << 16), ~(1ULL << 17), ~(1ULL << 18), ~(1ULL << 19), ~(1ULL << 20), ~(1ULL << 21), ~(1ULL << 22), ~(1ULL << 23),
    ~(1ULL << 24), ~(1ULL << 25), ~(1ULL << 26), ~(1ULL << 27), ~(1ULL << 28), ~(1ULL << 29), ~(1ULL << 30), ~(1ULL << 31),
    ~(1ULL << 32), ~(1ULL << 33), ~(1ULL << 34), ~(1ULL << 35), ~(1ULL << 36), ~(1ULL << 37), ~(1ULL << 38), ~(1ULL << 39),
    ~(1ULL << 40), ~(1ULL << 41), ~(1ULL << 42), ~(1ULL << 43), ~(1ULL << 44), ~(1ULL << 45), ~(1ULL << 46), ~(1ULL << 47),
    ~(1ULL << 48), ~(1ULL << 49), ~(1ULL << 50), ~(1ULL << 51), ~(1ULL << 52), ~(1ULL << 53), ~(1ULL << 54), ~(1ULL << 55),
    ~(1ULL << 56), ~(1ULL << 57), ~(1ULL << 58), ~(1ULL << 59), ~(1ULL << 60), ~(1ULL << 61), ~(1ULL << 62), ~(1ULL << 63),
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL,
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL,
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL,
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL,
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL,
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL,
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL,
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL
};

static inline void update_md(uint64_t *md, uint8_t index) {
  uint64_t carry = (md[0] >> 63) & carry_pdep_table[index];
  md[1] = _pdep_u64(md[1], high_order_pdep_table[index]) | carry;
  md[0] = _pdep_u64(md[0], low_order_pdep_table[index]);
}

static inline void update_tags_512(vqf_block *block, uint8_t index, uint8_t tag) {
  index -= 16;
  memmove(&block->tags[index + 1], &block->tags[index], sizeof(block->tags) / sizeof(block->tags[0]) - index - 1);
  block->tags[index] = tag;
}

static inline int word_rank(uint64_t val) {
  return __builtin_popcountll(val);
}

static inline uint64_t lookup_128(uint64_t *vector, uint64_t rank) {
   uint64_t lower_word = vector[0];
   uint64_t lower_rank = word_rank(lower_word);
   uint64_t lower_return = _pdep_u64(one[rank], lower_word) >> rank << sizeof(__uint128_t);
   int64_t higher_rank = (int64_t)rank - lower_rank;
   uint64_t higher_word = vector[1];
   uint64_t higher_return = _pdep_u64(one[higher_rank], higher_word);
   higher_return <<= (64 + sizeof(__uint128_t) - rank);
   return lower_return + higher_return;
}

static inline int64_t select_128(uint64_t *vector, uint64_t rank) {
   return _tzcnt_u64(lookup_128(vector, rank));
}



CVQFBitsBuilder::CVQFBitsBuilder(const size_t bits_per_key,
                                 const size_t num_probes, const uint64_t nslots)
    : bits_per_key_(bits_per_key), num_probes_(num_probes), nslots_(1ULL << nslots){
//  assert(bits_per_key_);
  assert(nslots_);

  total_blocks = (nslots_ + QUQU_SLOTS_PER_BLOCK)/QUQU_SLOTS_PER_BLOCK;
  total_size_in_bytes = sizeof(vqf_block) * total_blocks;

  filter_ = (vqf_filter *)malloc(sizeof(*filter_) + total_size_in_bytes);
  assert(filter_);

  filter_->metadata.total_size_in_bytes = total_size_in_bytes;
  filter_->metadata.nslots = total_blocks * QUQU_SLOTS_PER_BLOCK;
  filter_->metadata.key_remainder_bits = 8;
  filter_->metadata.range = total_blocks * QUQU_BUCKETS_PER_BLOCK * (1ULL << filter_->metadata.key_remainder_bits);
  filter_->metadata.nblocks = total_blocks;

  for (uint64_t  i = 0; i < total_blocks; i++) {
    filter_->blocks[i].md[0] = UINT64_MAX;
    filter_->blocks[i].md[1] = UINT64_MAX;
    filter_->blocks[i].md[1] = filter_->blocks[i].md[1] & ~(1ULL << 63);
  }
//  PrintFilter();
//  printf("[CYDBG] CVQFBitsBuilder\n");
  }

  CVQFBitsBuilder::~CVQFBitsBuilder() {}

  void CVQFBitsBuilder::AddKey(const Slice& key) {
    uint32_t hash = BloomHash(key);
    /*CYDBG cvqf_block_test*/
    if (hash_entries_.size() == 0 || hash != hash_entries_.back()) {
      hash_entries_.push_back(hash);
    }
    /*CYDBG*/

    vqf_metadata* metadata = &filter_->metadata;
    vqf_block* blocks = filter_->blocks;
    uint64_t key_remainder_bits = metadata->key_remainder_bits;
    uint64_t range = metadata->range;

    uint64_t temp64 = (uint64_t) hash;
    uint64_t hash64 = (temp64 | (temp64 << 32)) % range;/*Hash needs to be in range*/

    uint64_t block_index = hash64 >> key_remainder_bits;

    uint64_t *block_md = blocks[block_index/QUQU_BUCKETS_PER_BLOCK].md;
    uint64_t block_free = GetBlockFreeSpace(block_md);
    uint64_t tag = hash64 & TAG_MASK;
    uint64_t alt_block_index = ((hash64 ^ (tag * 0x5bd1e995)) % range) >> key_remainder_bits;
    __builtin_prefetch(&blocks[alt_block_index/QUQU_BUCKETS_PER_BLOCK]);

    if (block_free < QUQU_CHECK_ALT && block_index/QUQU_BUCKETS_PER_BLOCK != alt_block_index/QUQU_BUCKETS_PER_BLOCK) {
      uint64_t *alt_block_md = blocks[alt_block_index/QUQU_BUCKETS_PER_BLOCK].md;
      uint64_t alt_block_free = GetBlockFreeSpace(alt_block_md);
    
      if (alt_block_free > block_free) {
        block_index = alt_block_index;
        block_md = alt_block_md;
        block_free = alt_block_free;
      } else if (block_free == QUQU_BUCKETS_PER_BLOCK) {
        PrintBlock(block_index / QUQU_BUCKETS_PER_BLOCK);
        PrintBlock(alt_block_index / QUQU_BUCKETS_PER_BLOCK);
        fprintf(stderr, "vqf filter is full.\n");
        //return false;
        return ;
      }

    } else {
      if (block_free == QUQU_BUCKETS_PER_BLOCK) {
        PrintBlock(block_index / QUQU_BUCKETS_PER_BLOCK);
        fprintf(stderr, "vqf filter is full, no alternative.\n");
        return;
      }
    }

    uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
    uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;

    uint64_t slot_index = select_128(block_md, offset);
    uint64_t select_index = slot_index + offset - sizeof(__uint128_t);

    uint64_t preslot_index; // yongjin
    if (offset != 0) {
      preslot_index = select_128(block_md, offset - 1);
    } else {
      preslot_index = QUQU_PRESLOT;
    }

    uint64_t target_index;
    uint64_t end_target_index;
    uint8_t temp_tag;

    if (preslot_index == slot_index) {
      target_index = slot_index;
      update_tags_512(&blocks[index], target_index, tag);
      update_md(block_md, select_index);
//      PrintBlock(index);
      return;
    }

    else {
      target_index = slot_index;

      // sorting //////////////////
      uint64_t i;
      for (i = preslot_index; i < slot_index; i++) {
        // candidate
        if (blocks[index].tags[i - QUQU_PRESLOT] >= tag) {
          // the first tag, no need to think
          if (i == preslot_index) {
            target_index = i;
            break;
          }
          // could be counter
          else if (blocks[index].tags[i - 1 - QUQU_PRESLOT] == 0) {
            // corner case, [0, counter], found
            if (i == preslot_index + 1) {
              target_index = i;
              break;
            }
            // other cases
            else {
              temp_tag = blocks[index].tags[i - 2 - QUQU_PRESLOT];

              // corner case, [0, 0, 255], found
              if (temp_tag == 0) {
                target_index = i;
                break;
              }

              // counter (TAG_BIT = 8)
              else {
                while(blocks[index].tags[i - QUQU_PRESLOT] == QUQU_MAX) i++;
                  if (blocks[index].tags[i - QUQU_PRESLOT] != temp_tag) i++;
                  continue;
              }
            }
          }

          // found
          else {
            target_index = i;
            break;
          }
        }
      }
    // sorting //////////////////
    }

    // if tag that is ">=" is found in [preslot_index ---------- slot_index)
    if (target_index < slot_index) {

      // need counter
      if (blocks[index].tags[target_index - QUQU_PRESLOT] == tag) {
        // just find the other tag in [preslot_index ---------- slot_index)
        end_target_index = target_index + 1;
        while (end_target_index < slot_index) {

          // (if end_target_index == slot_index, there is no match)
          if (blocks[index].tags[end_target_index - QUQU_PRESLOT] == tag) break;
            end_target_index++;
        }

        // no extra match, just put it
        if (end_target_index == slot_index) {
          update_tags_512(&blocks[index], target_index, tag);
          update_md(block_md, select_index);
//          PrintBlock(index);
          return;
        }

        // counter //////////////////

        // extra match, tag is 0
        else if (tag == 0) {
          // [0, 0, ...]
          if (end_target_index == target_index + 1) {
            // check if [0, 0, 0, ...]
            if (end_target_index < slot_index - 1) {
              if (blocks[index].tags[end_target_index + 1 - QUQU_PRESLOT] == tag) {
               update_tags_512(&blocks[index], end_target_index, 1);
               update_md(block_md, select_index);
//               PrintBlock(index);
               return;
              }
              else {
                update_tags_512(&blocks[index], end_target_index, tag);
                update_md(block_md, select_index);
//                PrintBlock(index);
                return;
              }
	          }
            // [0, 0]
            else {
              update_tags_512(&blocks[index], target_index, tag);
              update_md(block_md, select_index);
//              PrintBlock(index);
              return;
	          }
          }

          // check if counter, [0, ... 0, 0, ...]
          else if (end_target_index < slot_index - 1) {
            if (blocks[index].tags[end_target_index + 1 - QUQU_PRESLOT] == tag) {

              // full counter
              if (blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] == QUQU_MAX) {
                update_tags_512(&blocks[index], end_target_index, 1);
                update_md(block_md, select_index);
//                PrintBlock(index);
                return;
              }

              // increment counter
              else {
                blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT]++;
//                PrintBlock(index);
                return;
              }
            }

            else {
              // wrong zero fetched
              update_tags_512(&blocks[index], target_index, tag);
              update_md(block_md, select_index);
//              PrintBlock(index);
              return;
            }
          }

          // wrong extra 0 fetched
          else {
            // [0 ... 0] ?
            printf("[CYDBG] ERROR\n");
            //update_tags_512(&blocks[index], target_index, tag);
            //update_md(block_md, select_index);
            return;
          }
        }

        // extra match, tag is 1
        else if (tag == 1) {

          // [1, 1], need to insert two tags
          if (end_target_index == target_index + 1) {

            // cannot insert two tags
            if (block_free == QUQU_BUCKETS_PER_BLOCK + 1) {
              printf("[CYDBG] ERROR\n");
              return;
            }

            // can insert two tags
            else {
              update_tags_512(&blocks[index], end_target_index, 0);
              update_md(block_md, select_index);
              update_tags_512(&blocks[index], end_target_index + 1, 2);
              update_md(block_md, select_index);
//              PrintBlock(index);
              return;
            }
          }

          // counter
          else if (blocks[index].tags[target_index + 1 - QUQU_PRESLOT] < tag) {

            // add new counter
            if (blocks[index].tags[end_target_index - 1  - QUQU_PRESLOT] == QUQU_MAX) {
              update_tags_512(&blocks[index], end_target_index, 2);
              update_md(block_md, select_index);
//              PrintBlock(index);
              return;
            }

            // increment counter
            else {
              blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT]++;
//              PrintBlock(index);
              return;
            }
          }

          // wrong extra 1 fetched
          else {
            update_tags_512(&blocks[index], target_index, tag);
            update_md(block_md, select_index);
            PrintBlock(index);
            return;
          }
        }

        // extra match, tag is 255
        else if (tag == QUQU_MAX) {
          // [255, 255]
          if (end_target_index == target_index + 1) {
            update_tags_512(&blocks[index], end_target_index, 1);
            update_md(block_md, select_index);
//            PrintBlock(index);
            return;
          }

          // [255, ... , 255]
          else {
            // add new counter
            if (blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] == QUQU_MAX - 1) {
              update_tags_512(&blocks[index], end_target_index, 1);
              update_md(block_md, select_index);
//              PrintBlock(index);
              return;
            }

            // increment counter
            else {
              blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT]++;
//              PrintBlock(index);
              return;
            }
          }
        }

        // extra match, other tags
        else {
          // [tag, tag]
          if (end_target_index == target_index + 1) {
            update_tags_512(&blocks[index], end_target_index, 1);
            update_md(block_md, select_index);
//            PrintBlock(index);
            return;
          }

         // counter
          else if (blocks[index].tags[target_index + 1 - QUQU_PRESLOT] < tag) {
            // add new counter
            if (blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] == QUQU_MAX) {
              update_tags_512(&blocks[index], end_target_index, 1);
              update_md(block_md, select_index);
//              PrintBlock(index);
              return;
            }

           // increment counter
            else {
              temp_tag = blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] + 1;
              if (temp_tag == tag) {
                temp_tag++;

                // need to put 0
                if (blocks[index].tags[target_index + 1 - QUQU_PRESLOT] != 0) {
                  blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] = temp_tag;
                  update_tags_512(&blocks[index], target_index + 1, 0);
                  update_md(block_md, select_index);
//                  PrintBlock(index);
                  return;
                }

                // no need to put 0
                else {
                  blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] = temp_tag;
//                  PrintBlock(index);
                  return;
                }
              } else {
                blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] = temp_tag;
//                PrintBlock(index);
                return ;
              }
            }
          }

          // wrong fetch
          else {
            update_tags_512(&blocks[index], target_index, tag);
            update_md(block_md, select_index);
//            PrintBlock(index);
            return;
          }
        }
        // counter //////////////////
      }

      // no need counter
      else {
        update_tags_512(&blocks[index], target_index, tag);
        update_md(block_md, select_index);
//        PrintBlock(index);
        return;
      }
    }

    // if not found in [preslot_index ---------- slot_index)
    else {
      update_tags_512(&blocks[index], target_index, tag); // slot_index
      update_md(block_md, select_index);
//      PrintBlock(index);
      return;
    }

    //something wrong
    printf("[CYDBG] Error\n");
    return;
  } //EndOf_AddKey

  Slice CVQFBitsBuilder::Finish(std::unique_ptr<const char[]>* buf) {
    hash_entries_.clear(); /*CYDBG cvqf_block_test*/
    const char* const_data = (char *)filter_;
    buf->reset(const_data);

    /*CYDBG changing filter to Slice*/
    size_t data_size = filter_->metadata.total_size_in_bytes + sizeof(vqf_metadata);
    char* data = new char[data_size];
    memset(data, 0, filter_->metadata.total_size_in_bytes);

    /*CYDBG filter_->metadata stored in data*/
    uint64_t index_num = 0;
    InsertData64ToChar(data, index_num, filter_->metadata.total_size_in_bytes);
    index_num += 8;
    InsertData64ToChar(data, index_num, filter_->metadata.key_remainder_bits);
    index_num += 8;
    InsertData64ToChar(data, index_num, filter_->metadata.range);
    index_num += 8;
    InsertData64ToChar(data, index_num, filter_->metadata.nblocks);
    index_num += 8;
    InsertData64ToChar(data, index_num, filter_->metadata.nslots);
    index_num += 8;

/*    printf("[CYDBG] Recovered Data\n");
    for (int i = 0; i < 5; i++) {
      uint64_t recovered_data = static_cast<uint64_t>(data[i] & 0x000000ff);
      printf("[CYDBG] recovered_data: %lx\n", recovered_data);
    }*/

/*    printf("[CYDBG] original data: %lx\n", filter_->metadata.total_size_in_bytes & 0x00000000000000ff);
    data[0] = static_cast<char>((filter_->metadata.total_size_in_bytes >> 0) & 0x00000000000000ff);
    uint64_t recovered_data = static_cast<uint64_t>(data[0] & 0x000000ff);
    printf("[CYDBG] recovered_data: %lx\n", recovered_data);*/
  
    for (uint64_t j = 0; j < filter_->metadata.nblocks; j++) {
      /*Insert vqf_block->md*/
      for (int i = 0; i < 2; i++) {
        InsertData64ToChar(data, index_num, filter_->blocks[j].md[i]);
        index_num += 8;
      }
      /*Insert vqf_block->tag*/
      for (int i = 0; i < 48; i++) {
        data[index_num] = filter_->blocks[j].tags[i];
        index_num ++;
      }
    }
//    printf("[CYDBF] CVQFBitsBuilder::Finish\n");
//    PrintFilter();
//    PrintBlock(1020);

/*    int block_index = 1020;
    printf("\n[CYDBG] data\n");
    printf("[CYDBG] filter->metadata[1]: ");
    for (int i = 64 * block_index + 55; i >= 40 + 64*block_index; i--) {
      printf("%x", (data[i] & 0x000000ff));
      if (i == 64 * block_index + 48)
        printf("\n[CYDBG] filter->metadata[0]: ");
    } 
    printf("\n[CYDBG] filter->tags: ");
    for (int i = 64 * block_index + 56; i < 64 * (block_index + 1) + 40; i++) {
    	printf("%x ", (data[i] & 0x000000ff));
    }
    printf("\n");*/


//    return Slice((std::string &)filter);
    return Slice(data, data_size);
  }

vqf_filter* CVQFBitsBuilder::GetFilter() {
  return filter_;
}

void CVQFBitsBuilder::InsertData64ToChar(char* data, int index, uint64_t input) {
  for (int i = index; i < index + 8; i++) {
    data[i] = static_cast<char>((input >> (8*i)) & 0x00000000000000ff);
  }
}

void CVQFBitsBuilder::PrintFilter() {
  //print metadata
  printf("cvqf_metadata: \n \
          total_size_in_bytes: %lx \n \
          key_remainder_bits: %lx \n \
          range: %lx \n \
          nblocks: %lx \n \
          nslots: %lx \n\n", \
          filter_->metadata.total_size_in_bytes, filter_->metadata.key_remainder_bits,\
          filter_->metadata.range, filter_->metadata.nblocks, filter_->metadata.nslots);

  //print block, md
/*  for (uint64_t  i = 1020; i < 1021; i++) {//total_blocks; i++) {
    printf("vqf_block: \n \
            blocks[%lu].md[0]: %lx\n \
            blocks[%lu].md[1]: %lx\n", i, filter_->blocks[i].md[0], i, filter_->blocks[i].md[1]);
  }

  //print block, tag
  printf("vqf_block: \n"); 
  for (uint64_t  i = 1020; i < 1021; i++) {//total_blocks; i++) {
    printf("           blocks[%lu].tags: ", i);
    for (uint64_t j = 0; j < 48; j++) {
      printf("%x ", filter_->blocks[i].tags[j]);
    }
    printf("\n");
  }*/
}

void CVQFBitsBuilder::PrintBits(__uint128_t num, int numbits) {
  int i;
  for (i = 0; i < numbits; i++) {
    if (i != 0 && i % 8 == 0) {
      printf(":");
    }
    printf("%d", ((num >> i) & 1) == 1);
  }
  puts("");
}


void CVQFBitsBuilder::PrintTags(uint8_t *tags, uint32_t size) {
  for (uint8_t i = 0; i < size; i++)
    printf("%d ", (uint32_t)tags[i]);
  printf("\n");
}

void CVQFBitsBuilder::PrintBlock(uint64_t block_index) {
  printf("block index: %ld\n", block_index);
  printf("metadata: ");
  uint64_t *md = filter_->blocks[block_index].md;
  PrintBits(*(__uint128_t *)md, QUQU_BUCKETS_PER_BLOCK + QUQU_SLOTS_PER_BLOCK);
  printf("tags: ");
  PrintTags(filter_->blocks[block_index].tags, QUQU_SLOTS_PER_BLOCK);
}

inline uint64_t CVQFBitsBuilder::GetBlockFreeSpace(uint64_t *vector) {
  uint64_t lower_word = vector[0];
  uint64_t higher_word = vector[1];
  return word_rank(lower_word) + word_rank(higher_word);
}


namespace {
class CVQFBitsReader : public FilterBitsReader {
 public:
  explicit CVQFBitsReader(const Slice& contents)
      : data_(const_cast<char*>(contents.data())) {
    assert(data_);
//    filter_ = (vqf_filter *)data_;
    uint64_t nslots_ = 1ULL << 17;
    uint64_t total_blocks = (nslots_ + QUQU_SLOTS_PER_BLOCK)/QUQU_SLOTS_PER_BLOCK;
    uint64_t total_size_in_bytes = sizeof(vqf_block) * total_blocks;

    filter_ = (vqf_filter *)calloc(sizeof(*filter_) + total_size_in_bytes, 1);

    /*CYDBG Load vqf_metadata*/
    for (int i = 0; i < 8; i++) {
      filter_->metadata.total_size_in_bytes |= (static_cast<uint64_t>(data_[i] & 0x000000ff) << (i*8));
      filter_->metadata.key_remainder_bits |= (static_cast<uint64_t>(data_[i+8] & 0x000000ff) << (i*8));
      filter_->metadata.range |= (static_cast<uint64_t>(data_[i+16] & 0x000000ff) << (i*8));
      filter_->metadata.nblocks |= (static_cast<uint64_t>(data_[i+24] & 0x000000ff) << (i*8));
      filter_->metadata.nslots |= (static_cast<uint64_t>(data_[i+32] & 0x000000ff) << (i*8));
    }
/*    printf("[CYDBG] CVQFBitsReader\nvqf_metadata: \n \
            total_size_in_bytes: %lx \n \
            key_remainder_bits: %lx \n \
            range: %lx \n \
            nblocks: %lx \n \
            nslots: %lx \n\n", \
            filter_->metadata.total_size_in_bytes, filter_->metadata.key_remainder_bits,\
            filter_->metadata.range, filter_->metadata.nblocks, filter_->metadata.nslots);*/

    /*CYDBG Load vqf_block*/
    for (uint64_t j = 0; j < filter_->metadata.nblocks; j++) {
      /*Insert vqf_block->md*/
      for (int i = 0; i < 8; i++) {
        filter_->blocks[j].md[0] |= (static_cast<uint64_t>(data_[j*64 + 40 +i] & 0x000000ff) << (i*8));
        filter_->blocks[j].md[1] |= (static_cast<uint64_t>(data_[j*64 + 48 + i] & 0x000000ff) << (i*8));
      }
      /*Insert vqf_block->tags*/
      for (int i = 0; i < 48; i++) {
        filter_->blocks[j].tags[i] = (static_cast<uint64_t>(data_[j*64 + 56 + i] & 0x000000ff));
      }
    }

//    PrintBlock(1020);

  }

  ~CVQFBitsReader() override {}

  static inline bool check_tags(vqf_filter* filter, uint64_t tag, uint64_t block_index) {
    uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
    uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;
    __m256i bcast = _mm256_set1_epi8(tag);
    __m256i block = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&filter->blocks[index]));
    __m256i result1t = _mm256_cmpeq_epi8(bcast, block);
    __mmask32 result1 = _mm256_movemask_epi8(result1t);
    block = _mm256_loadu_si256(reinterpret_cast<__m256i*>((uint8_t*)&filter->blocks[index]+32));
    __m256i result2t = _mm256_cmpeq_epi8(bcast, block);
    __mmask32 result2 = _mm256_movemask_epi8(result2t);
    uint64_t result = (uint64_t)result2 << 32 | (uint64_t)result1;

    if (result == 0) {
      // no matching tags, can bail
      return false;
    }

    uint64_t start = offset != 0 ? lookup_128(filter->blocks[index].md, offset - 1) : one[0] << 2 * sizeof(uint64_t); // 1 << 16
    uint64_t end = lookup_128(filter->blocks[index].md, offset);
    uint64_t mask = end - start;
    vqf_block * blocks = filter->blocks;
    uint64_t equalLocations = mask & result;
    uint64_t slot_start = _tzcnt_u64(start);
    uint64_t slot_end = _tzcnt_u64(end);
    uint64_t slot_check;

     // 255 should be last tag
    if (tag == QUQU_MAX) {
      if (((equalLocations >> (slot_end - 1)) & 1 ) == 1) {
        return true;
      }
      else {
        return false;
      }
    }

    // 0 should be first tag
    else if (tag == 0) {
      if (((equalLocations >> slot_start) & 1 ) == 1) {
        return true;
      }
      else {
        return false;
      }
    }

    // other tags
    else {
      // filter->blocks[index].tags[slot_check - 16];
      while (equalLocations != 0) {
        // only check necessaries
        slot_check = _tzcnt_u64(equalLocations);

        // if first
        if (slot_check == slot_start) {
          return true;
        }

        // if last
        else if (slot_check == slot_end - 1) {
          return true;
        }

        // not first, nor last
        else {
        // the escape sequence
          if (blocks[index].tags[slot_check - 1 - QUQU_PRESLOT] > tag) {
            // counter
          }

          // [... 0, tag ...]
          else if (blocks[index].tags[slot_check - 1 - QUQU_PRESLOT] == 0) {

            // [0, tag ...]
            if (slot_check == slot_start + 1) {
              if (slot_check < slot_end - 2) {
                // [0, tag, 0, 0 ...]
                if (blocks[index].tags[slot_check + 1 - QUQU_PRESLOT] == 0 && blocks[index].tags[slot_check + 2 - QUQU_PRESLOT] == 0) {
                  // counter
                }

                // not [0, tag, 0, 0 ...] sequence
                else {
                  return true;
                }
              }

              // cannot even make the sequence
              else {
                return true;
              }
            }

            // [... 0, tag ...]
            else {
              // [ ... 0, 0, tag ...]
              if (blocks[index].tags[slot_check - 2 - QUQU_PRESLOT] == 0) {
                return true;
              }
              else {
                // counter
              }
            }
          }

          // tag before is less than
          else if (blocks[index].tags[slot_check - 1 - QUQU_PRESLOT] < tag) {
            return true;
          }

          // tag before is equal to
          else {
          }
        }
        equalLocations &= ~(one[0] << slot_check);
      }
    }
      return false;
  }

  bool MayMatch(const Slice& entry) override {
    uint32_t hash = BloomHash(entry);

    vqf_metadata* metadata = &filter_->metadata;
    uint64_t key_remainder_bits = metadata->key_remainder_bits;
    uint64_t range = metadata->range;

    uint64_t temp64 = (uint64_t) hash;
    uint64_t hash64 = (temp64 | (temp64 << 32)) % range;/*Hash needs to be in range*/
//    uint64_t hash64 = hash % range;/*Hash needs to be in range*/
    uint64_t tag = hash64 & TAG_MASK;
    uint64_t block_index = hash64 >> key_remainder_bits;
    uint64_t alt_block_index = ((hash64 ^(tag * 0x5bd1e995)) % range) >> key_remainder_bits;

    __builtin_prefetch(&filter_->blocks[alt_block_index / QUQU_BUCKETS_PER_BLOCK]);
 
    bool result =  check_tags(filter_, tag, block_index) || check_tags(filter_, tag, alt_block_index);
//    printf("[CYDBG] result: %d\n", result);
    return result;
    /*CVQF*/
  }

  /*PrintFunctions*/
  void PrintBits(__uint128_t num, int numbits) {
    int i;
    for (i = 0; i < numbits; i++) {
      if (i != 0 && i % 8 == 0) {
        printf(":");
      }
      printf("%d", ((num >> i) & 1) == 1);
    }
    puts("");
  }

  void PrintTags(uint8_t *tags, uint32_t size) {
    for (uint8_t i = 0; i < size; i++)
      printf("%d ", (uint32_t)tags[i]);
    printf("\n");
  }

  void PrintBlock(uint64_t block_index) {
    printf("block index: %ld\n", block_index);
    printf("metadata: ");
    uint64_t *md = filter_->blocks[block_index].md;
    PrintBits(*(__uint128_t *)md, QUQU_BUCKETS_PER_BLOCK + QUQU_SLOTS_PER_BLOCK);
    printf("tags: ");
    PrintTags(filter_->blocks[block_index].tags, QUQU_SLOTS_PER_BLOCK);
  }

 private:
  // Filter meta data
  char* data_;
  vqf_filter* filter_;

  // No Copy allowed
  CVQFBitsReader(const CVQFBitsReader&);
  void operator=(const CVQFBitsReader&);
};

// An implementation of filter policy
class CVQFPolicy : public FilterPolicy {
 public:
  explicit CVQFPolicy(int bits_per_key, bool use_block_based_builder, uint64_t nslots)
      : bits_per_key_(bits_per_key), hash_func_(BloomHash),
        use_block_based_builder_(use_block_based_builder), nslots_(nslots) {
    initialize();
  }

  ~CVQFPolicy() override {}

  const char* Name() const override { return "rocksdb.BuiltinCVQF"; }

  void CreateFilter(const Slice* keys, int n, std::string* dst) const override {
    // Compute bloom filter size (in both bits and bytes)
    size_t bits = n * bits_per_key_;

    // For small n, we can see a very high false positive rate.  Fix it
    // by enforcing a minimum bloom filter length.
    if (bits < 64) bits = 64;

    size_t bytes = (bits + 7) / 8;
    bits = bytes * 8;

    const size_t init_size = dst->size();
    dst->resize(init_size + bytes, 0);
    dst->push_back(static_cast<char>(num_probes_));  // Remember # of probes
    char* array = &(*dst)[init_size];
    for (size_t i = 0; i < (size_t)n; i++) {
      // Use double-hashing to generate a sequence of hash values.
      // See analysis in [Kirsch,Mitzenmacher 2006].
      uint32_t h = hash_func_(keys[i]);
      const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
      for (size_t j = 0; j < num_probes_; j++) {
        const uint32_t bitpos = h % bits;
        array[bitpos/8] |= (1 << (bitpos % 8));
        h += delta;
      }
    }
  }

  bool KeyMayMatch(const Slice& key, const Slice& bloom_filter) const override {
    const size_t len = bloom_filter.size();
    if (len < 2) return false;

    const char* array = bloom_filter.data();
    const size_t bits = (len - 1) * 8;

    // Use the encoded k so that we can read filters generated by
    // bloom filters created using different parameters.
    const size_t k = array[len-1];
    if (k > 30) {
      // Reserved for potentially new encodings for short bloom filters.
      // Consider it a match.
      return true;
    }

    uint32_t h = hash_func_(key);
    const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    for (size_t j = 0; j < k; j++) {
      const uint32_t bitpos = h % bits;
      if ((array[bitpos/8] & (1 << (bitpos % 8))) == 0) return false;
      h += delta;
    }
    return true;
  }

  FilterBitsBuilder* GetFilterBitsBuilder() const override {
    if (use_block_based_builder_) {
      return nullptr;
    }

    return new CVQFBitsBuilder(bits_per_key_, num_probes_, nslots_); 
  }

  FilterBitsReader* GetFilterBitsReader(const Slice& contents) const override {
    return new CVQFBitsReader(contents);
  }

  // If choose to use block based builder
  bool UseBlockBasedBuilder() { return use_block_based_builder_; }

 private:
  size_t bits_per_key_;
  size_t num_probes_;
  uint32_t (*hash_func_)(const Slice& key);

  const bool use_block_based_builder_;
  uint64_t nslots_;

  void initialize() {
    // We intentionally round down to reduce probing cost a little bit
    num_probes_ = static_cast<size_t>(bits_per_key_ * 0.69);  // 0.69 =~ ln(2)
    if (num_probes_ < 1) num_probes_ = 1;
    if (num_probes_ > 30) num_probes_ = 30;
  }
};

}  // namespace

const FilterPolicy* NewCVQFPolicy(int bits_per_key,
                                  bool use_block_based_builder, uint64_t nslots) {
  return new CVQFPolicy(bits_per_key, use_block_based_builder, nslots);
}

}  // namespace rocksdb
