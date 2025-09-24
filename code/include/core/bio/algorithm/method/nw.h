#pragma once
#ifndef CORE_BIO_ALGORITHM_METHOD_NW_H
#define CORE_BIO_ALGORITHM_METHOD_NW_H

#include "core/bio/types.h"

score_t align_nw(const sequence_ptr_t seq1, const sequence_ptr_t seq2);

#endif // CORE_BIO_ALGORITHM_METHOD_NW_H