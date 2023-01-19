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

  filter = (vqf_filter *)malloc(sizeof(*filter) + total_size_in_bytes);
  assert(filter);

  filter->metadata.total_size_in_bytes = total_size_in_bytes;
  filter->metadata.nslots = total_blocks * QUQU_SLOTS_PER_BLOCK;
  filter->metadata.key_remainder_bits = 8;
  filter->metadata.range = total_blocks * QUQU_BUCKETS_PER_BLOCK * (1ULL << filter->metadata.key_remainder_bits);
  filter->metadata.nblocks = total_blocks;

  for (uint64_t  i = 0; i < total_blocks; i++) {
    filter->blocks[i].md[0] = UINT64_MAX;
    filter->blocks[i].md[1] = UINT64_MAX;
    filter->blocks[i].md[1] = filter->blocks[i].md[1] & ~(1ULL << 63);
  }
  }

  CVQFBitsBuilder::~CVQFBitsBuilder() {}

  void CVQFBitsBuilder::AddKey(const Slice& key) {
    uint32_t hash = BloomHash(key);
//    uint64_t hash64 = ((uint64_t)hash << 32) | (uint64_t)hash;
//    uint64_t hash64 = (uint64_t) hash;
//    printf("hash64: %lx", hash64);

    vqf_metadata* metadata = &filter->metadata;
    vqf_block* blocks = filter->blocks;
    uint64_t key_remainder_bits = metadata->key_remainder_bits;
    uint64_t range = metadata->range;

    uint64_t hash64 = hash % range;/*Hash needs to be in range*/

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
      PrintBlock(index);
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
          PrintBlock(index);
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
               PrintBlock(index);
               return;
              }
              else {
                update_tags_512(&blocks[index], end_target_index, tag);
                update_md(block_md, select_index);
                PrintBlock(index);
                return;
              }
	          }
            // [0, 0]
            else {
              update_tags_512(&blocks[index], target_index, tag);
              update_md(block_md, select_index);
              PrintBlock(index);
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
                PrintBlock(index);
                return;
              }

              // increment counter
              else {
                blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT]++;
                PrintBlock(index);
                return;
              }
            }

            else {
              // wrong zero fetched
              update_tags_512(&blocks[index], target_index, tag);
              update_md(block_md, select_index);
              PrintBlock(index);
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
              PrintBlock(index);
              return;
            }
          }

          // counter
          else if (blocks[index].tags[target_index + 1 - QUQU_PRESLOT] < tag) {

            // add new counter
            if (blocks[index].tags[end_target_index - 1  - QUQU_PRESLOT] == QUQU_MAX) {
              update_tags_512(&blocks[index], end_target_index, 2);
              update_md(block_md, select_index);
              PrintBlock(index);
              return;
            }

            // increment counter
            else {
              blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT]++;
              PrintBlock(index);
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
            PrintBlock(index);
            return;
          }

          // [255, ... , 255]
          else {
            // add new counter
            if (blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] == QUQU_MAX - 1) {
              update_tags_512(&blocks[index], end_target_index, 1);
              update_md(block_md, select_index);
              PrintBlock(index);
              return;
            }

            // increment counter
            else {
              blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT]++;
              PrintBlock(index);
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
            PrintBlock(index);
            return;
          }

         // counter
          else if (blocks[index].tags[target_index + 1 - QUQU_PRESLOT] < tag) {
            // add new counter
            if (blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] == QUQU_MAX) {
              update_tags_512(&blocks[index], end_target_index, 1);
              update_md(block_md, select_index);
              PrintBlock(index);
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
                  PrintBlock(index);
                  return;
                }

                // no need to put 0
                else {
                  blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] = temp_tag;
                  PrintBlock(index);
                  return;
                }
              } else {
                blocks[index].tags[end_target_index - 1 - QUQU_PRESLOT] = temp_tag;
                PrintBlock(index);
                return ;
              }
            }
          }

          // wrong fetch
          else {
            update_tags_512(&blocks[index], target_index, tag);
            update_md(block_md, select_index);
            PrintBlock(index);
            return;
          }
        }
        // counter //////////////////
      }

      // no need counter
      else {
        update_tags_512(&blocks[index], target_index, tag);
        update_md(block_md, select_index);
        PrintBlock(index);
        return;
      }
    }

    // if not found in [preslot_index ---------- slot_index)
    else {
      update_tags_512(&blocks[index], target_index, tag); // slot_index
      update_md(block_md, select_index);
      PrintBlock(index);
      return;
    }

    //something wrong
    printf("[CYDBG] Error\n");
    return;
  } //EndOf_AddKey

  Slice CVQFBitsBuilder::Finish(std::unique_ptr<const char[]>* buf) {
    uint32_t total_bits, num_lines;
    char* data = ReserveSpace(static_cast<int>(hash_entries_.size()),
                              &total_bits, &num_lines);
    assert(data);

    if (total_bits != 0 && num_lines != 0) {
      for (auto h : hash_entries_) {
        AddHash(h, data, num_lines, total_bits);
      }
    }
    data[total_bits/8] = static_cast<char>(num_probes_);
    EncodeFixed32(data + total_bits/8 + 1, static_cast<uint32_t>(num_lines));

    const char* const_data = data;
    buf->reset(const_data);
    hash_entries_.clear();

    return Slice(data, total_bits / 8 + 5);
  }

vqf_filter* CVQFBitsBuilder::GetFilter() {
  return filter;
}

void CVQFBitsBuilder::PrintFilter() {
  //print metadata
  printf("vqf_metadata: \n \
          total_size_in_bytes: %lu \n \
          key_remainder_bits: %lu \n \
          range: %lu \n \
          nblocks: %lu \n \
          nslots: %lu \n\n", \
          filter->metadata.total_size_in_bytes, filter->metadata.key_remainder_bits,\
          filter->metadata.range, filter->metadata.nblocks, filter->metadata.nslots);

  //print block, md
  for (uint64_t  i = 0; i < 10; i++) {//total_blocks; i++) {
    printf("vqf_block: \n \
            blocks[%lu].md[0]: %lx\n \
            blocks[%lu].md[1]: %lx\n", i, filter->blocks[i].md[0], i, filter->blocks[i].md[1]);
  }

  //print bock, tag
  printf("vqf_block: \n"); 
  for (uint64_t  i = 0; i < 10; i++) {//total_blocks; i++) {
    for (uint64_t j = 0; j < 48; j++) {
      printf("            blocks[%lu].tags[%lu]: %u\n", i, j, filter->blocks[i].tags[j]);
    }
  }
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
  uint64_t *md = filter->blocks[block_index].md;
  PrintBits(*(__uint128_t *)md, QUQU_BUCKETS_PER_BLOCK + QUQU_SLOTS_PER_BLOCK);
  printf("tags: ");
  PrintTags(filter->blocks[block_index].tags, QUQU_SLOTS_PER_BLOCK);
}

uint32_t CVQFBitsBuilder::GetTotalBitsForLocality(uint32_t total_bits) {
  uint32_t num_lines =
      (total_bits + CACHE_LINE_SIZE * 8 - 1) / (CACHE_LINE_SIZE * 8);

  // Make num_lines an odd number to make sure more bits are involved
  // when determining which block.
  if (num_lines % 2 == 0) {
    num_lines++;
  }
  return num_lines * (CACHE_LINE_SIZE * 8);
}

uint32_t CVQFBitsBuilder::CalculateSpace(const int num_entry,
                                               uint32_t* total_bits,
                                               uint32_t* num_lines) {
  assert(bits_per_key_);
  if (num_entry != 0) {
    uint32_t total_bits_tmp = num_entry * static_cast<uint32_t>(bits_per_key_);

    *total_bits = GetTotalBitsForLocality(total_bits_tmp);
    *num_lines = *total_bits / (CACHE_LINE_SIZE * 8);
    assert(*total_bits > 0 && *total_bits % 8 == 0);
  } else {
    // filter is empty, just leave space for metadata
    *total_bits = 0;
    *num_lines = 0;
  }

  // Reserve space for Filter
  uint32_t sz = *total_bits / 8;
  sz += 5;  // 4 bytes for num_lines, 1 byte for num_probes
  return sz;
}

char* CVQFBitsBuilder::ReserveSpace(const int num_entry,
                                          uint32_t* total_bits,
                                          uint32_t* num_lines) {
  uint32_t sz = CalculateSpace(num_entry, total_bits, num_lines);
  char* data = new char[sz];
  memset(data, 0, sz);
  return data;
}

int CVQFBitsBuilder::CalculateNumEntry(const uint32_t space) {
  assert(bits_per_key_);
  assert(space > 0);
  uint32_t dont_care1, dont_care2;
  int high = (int) (space * 8 / bits_per_key_ + 1);
  int low = 1;
  int n = high;
  for (; n >= low; n--) {
    uint32_t sz = CalculateSpace(n, &dont_care1, &dont_care2);
    if (sz <= space) {
      break;
    }
  }
  assert(n < high);  // High should be an overestimation
  return n;
}

inline void CVQFBitsBuilder::AddHash(uint32_t h, char* data,
    uint32_t num_lines, uint32_t total_bits) {
#ifdef NDEBUG
  (void)total_bits;
#endif
  assert(num_lines > 0 && total_bits > 0);

  const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
  uint32_t b = (h % num_lines) * (CACHE_LINE_SIZE * 8);

  for (uint32_t i = 0; i < num_probes_; ++i) {
    // Since CACHE_LINE_SIZE is defined as 2^n, this line will be optimized
    // to a simple operation by compiler.
    const uint32_t bitpos = b + (h % (CACHE_LINE_SIZE * 8));
    data[bitpos / 8] |= (1 << (bitpos % 8));

    h += delta;
  }
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
      : data_(const_cast<char*>(contents.data())),
        data_len_(static_cast<uint32_t>(contents.size())),
        num_probes_(0),
        num_lines_(0),
        log2_cache_line_size_(0) {
    assert(data_);
    GetFilterMeta(contents, &num_probes_, &num_lines_);
    // Sanitize broken parameter
    if (num_lines_ != 0 && (data_len_-5) % num_lines_ != 0) {
      num_lines_ = 0;
      num_probes_ = 0;
    } else if (num_lines_ != 0) {
      while (true) {
        uint32_t num_lines_at_curr_cache_size =
            (data_len_ - 5) >> log2_cache_line_size_;
        if (num_lines_at_curr_cache_size == 0) {
          // The cache line size seems not a power of two. It's not supported
          // and indicates a corruption so disable using this filter.
          assert(false);
          num_lines_ = 0;
          num_probes_ = 0;
          break;
        }
        if (num_lines_at_curr_cache_size == num_lines_) {
          break;
        }
        ++log2_cache_line_size_;
      }
    }
  }

  ~CVQFBitsReader() override {}

  bool MayMatch(const Slice& entry) override {
    if (data_len_ <= 5) {   // remain same with original filter
      return false;
    }
    // Other Error params, including a broken filter, regarded as match
    if (num_probes_ == 0 || num_lines_ == 0) return true;
    uint32_t hash = BloomHash(entry);
    return HashMayMatch(hash, Slice(data_, data_len_),
                        num_probes_, num_lines_);
  }

 private:
  // Filter meta data
  char* data_;
  uint32_t data_len_;
  size_t num_probes_;
  uint32_t num_lines_;
  uint32_t log2_cache_line_size_;

  // Get num_probes, and num_lines from filter
  // If filter format broken, set both to 0.
  void GetFilterMeta(const Slice& filter, size_t* num_probes,
                             uint32_t* num_lines);

  // "filter" contains the data appended by a preceding call to
  // FilterBitsBuilder::Finish. This method must return true if the key was
  // passed to FilterBitsBuilder::AddKey. This method may return true or false
  // if the key was not on the list, but it should aim to return false with a
  // high probability.
  //
  // hash: target to be checked
  // filter: the whole filter, including meta data bytes
  // num_probes: number of probes, read before hand
  // num_lines: filter metadata, read before hand
  // Before calling this function, need to ensure the input meta data
  // is valid.
  bool HashMayMatch(const uint32_t& hash, const Slice& filter,
      const size_t& num_probes, const uint32_t& num_lines);

  // No Copy allowed
  CVQFBitsReader(const CVQFBitsReader&);
  void operator=(const CVQFBitsReader&);
};

void CVQFBitsReader::GetFilterMeta(const Slice& filter,
    size_t* num_probes, uint32_t* num_lines) {
  uint32_t len = static_cast<uint32_t>(filter.size());
  if (len <= 5) {
    // filter is empty or broken
    *num_probes = 0;
    *num_lines = 0;
    return;
  }

  *num_probes = filter.data()[len - 5];
  *num_lines = DecodeFixed32(filter.data() + len - 4);
}

bool CVQFBitsReader::HashMayMatch(const uint32_t& hash,
    const Slice& filter, const size_t& num_probes,
    const uint32_t& num_lines) {
  uint32_t len = static_cast<uint32_t>(filter.size());
  if (len <= 5) return false;  // remain the same with original filter

  // It is ensured the params are valid before calling it
  assert(num_probes != 0);
  assert(num_lines != 0 && (len - 5) % num_lines == 0);
  const char* data = filter.data();

  uint32_t h = hash;
  const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
  // Left shift by an extra 3 to convert bytes to bits
  uint32_t b = (h % num_lines) << (log2_cache_line_size_ + 3);
  PREFETCH(&data[b / 8], 0 /* rw */, 1 /* locality */);
  PREFETCH(&data[b / 8 + (1 << log2_cache_line_size_) - 1], 0 /* rw */,
           1 /* locality */);

  for (uint32_t i = 0; i < num_probes; ++i) {
    // Since CACHE_LINE_SIZE is defined as 2^n, this line will be optimized
    //  to a simple and operation by compiler.
    const uint32_t bitpos = b + (h & ((1 << (log2_cache_line_size_ + 3)) - 1));
    if (((data[bitpos / 8]) & (1 << (bitpos % 8))) == 0) {
      return false;
    }

    h += delta;
  }

  return true;
}

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
