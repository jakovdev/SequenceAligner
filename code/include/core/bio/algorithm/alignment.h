#pragma once
#ifndef CORE_BIO_ALGORITHM_ALIGNMENT_H
#define CORE_BIO_ALGORITHM_ALIGNMENT_H

#include "core/bio/types.h"

score_t align_pairwise(const sequence_ptr_t seq1, const sequence_ptr_t seq2);

#endif // CORE_BIO_ALGORITHM_ALIGNMENT_H