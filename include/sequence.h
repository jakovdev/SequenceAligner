#ifndef SEQUENCE_H
#define SEQUENCE_H

#include "common.h"

typedef struct {
    char* data;
    size_t length;
} Sequence;

#define SEQ_POOL_BLOCK_SIZE (4 * MiB)

typedef struct SeqMemBlock {
    char* data;
    size_t used;
    size_t capacity;
    struct SeqMemBlock* next;
} SeqMemBlock;

typedef struct {
    SeqMemBlock* head;
    SeqMemBlock* current;
    size_t total_bytes;
    size_t total_allocated;
    size_t block_count;
} SeqMemPool;

static SeqMemPool g_seq_pool = {0};

INLINE void init_seq_pool(void) {
    if (g_seq_pool.head != NULL) return;
    
    g_seq_pool.head = (SeqMemBlock*)malloc(sizeof(SeqMemBlock));
    if (!g_seq_pool.head) return;
    
    g_seq_pool.head->data = (char*)huge_page_alloc(SEQ_POOL_BLOCK_SIZE);
    if (!g_seq_pool.head->data) {
        free(g_seq_pool.head);
        g_seq_pool.head = NULL;
        return;
    }
    
    g_seq_pool.head->used = 0;
    g_seq_pool.head->capacity = SEQ_POOL_BLOCK_SIZE;
    g_seq_pool.head->next = NULL;
    g_seq_pool.current = g_seq_pool.head;
    g_seq_pool.total_bytes = SEQ_POOL_BLOCK_SIZE;
    g_seq_pool.total_allocated = 0;
    g_seq_pool.block_count = 1;
}

INLINE char* alloc_from_pool(size_t size) {
    if (g_seq_pool.head == NULL) {
        init_seq_pool();
        if (g_seq_pool.head == NULL) return NULL;
    }
    
    size = (size + 7) & ~7;
    
    if (g_seq_pool.current->used + size > g_seq_pool.current->capacity) {
        size_t new_block_size = SEQ_POOL_BLOCK_SIZE;
        if (size > new_block_size) {
            new_block_size = size;
        }
        
        SeqMemBlock* new_block = (SeqMemBlock*)malloc(sizeof(SeqMemBlock));
        if (!new_block) return NULL;
        
        new_block->data = (char*)malloc(new_block_size);
        if (!new_block->data) {
            free(new_block);
            return NULL;
        }
        
        new_block->used = 0;
        new_block->capacity = new_block_size;
        new_block->next = NULL;
        
        g_seq_pool.current->next = new_block;
        g_seq_pool.current = new_block;
        g_seq_pool.total_bytes += new_block_size;
        g_seq_pool.block_count++;
    }
    
    char* result = g_seq_pool.current->data + g_seq_pool.current->used;
    g_seq_pool.current->used += size;
    g_seq_pool.total_allocated += size;
    
    return result;
}

INLINE void free_seq_pool(void) {
    if (g_seq_pool.head == NULL) return;
    
    SeqMemBlock* block = g_seq_pool.head;
    while (block) {
        SeqMemBlock* next = block->next;
        aligned_free(block->data);
        free(block);
        block = next;
    }
    g_seq_pool.head = NULL;
    g_seq_pool.current = NULL;
    g_seq_pool.total_bytes = 0;
    g_seq_pool.total_allocated = 0;
    g_seq_pool.block_count = 0;
}

INLINE void init_sequence(Sequence* seq, const char* data, size_t length) {
    seq->length = length;
    
    seq->data = alloc_from_pool(length + 1);
    if (seq->data) {
        memcpy(seq->data, data, length);
        seq->data[length] = '\0';
    }
}

#endif // SEQUENCE_H