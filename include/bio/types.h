#ifndef BIO_TYPES_H
#define BIO_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "system/types.h"

/* NOTE: Additional alignment methods can be added here if needed.
 *       However, this requires implementing the corresponding algorithm.
 */
enum AlignmentMethod {
	ALIGN_INVALID = -1,
	ALIGN_GOTOH_AFFINE,
	ALIGN_NEEDLEMAN_WUNSCH,
	ALIGN_SMITH_WATERMAN,
	/* NOTE: EXPANDABLE enum AlignmentMethod */
	ALIGN_COUNT
};

extern enum AlignmentMethod METHOD;

extern s32 GAP_PEN;
extern s32 GAP_OPEN;
extern s32 GAP_EXT;

#ifdef __cplusplus
}
#endif

#endif /* BIO_TYPES_H */
