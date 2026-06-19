#include "io/source.h"

#include <ctype.h>
#include <limits.h>
#include <print.h>
#include <string.h>

#include "bio/align.h"
#include "io/input.h"
#include "util/macros.h"

static const char *EXTS[] = {
	"fasta", "fa", "fas", "fna", "ffn", "faa", "frn", "mpfa", nullptr,
};

static enum parse_result parse_fasta(struct source src, struct input *in)
{
	const char **ext = EXTS;
	for (; *ext; ext++) {
		if (strcasecmp(*ext, src.ext) == 0)
			break;
	}
	if (!*ext)
		return PARSER_UNSUPPORTED;

	const uchar *p = src.file;
	if (*p != '>') {
		perr("Data before first header");
		return PARSER_ERROR;
	}

	s32 num = 0;
	s32 max = 0;
	s64 sum = 0;
	uchar *w = src.file;
	while (p < src.fend) {
		while (p < src.fend && *p != '\n' && *p != '\r')
			p++;
		while (p < src.fend && (*p == '\n' || *p == '\r'))
			p++;
		if (p >= src.fend) {
			perr("Last header has no data");
			return PARSER_ERROR;
		}

		num++;
		s32 slen = 0;
		while (p < src.fend && *p != '>') {
			uchar c = (uchar)toupper(*p++);
			if (c == '\r' || c == '\n' || c == ' ')
				continue;
			if (c == '\0' || c > SCHAR_MAX) {
				perr("Sequence #%d is corrupted", num);
				return PARSER_ERROR;
			}
			if (SEQ_LUT[c] < 0) {
				perr("Sequence #%d is invalid", num);
				return PARSER_ERROR;
			}
			*w++ = c;
			slen++;
		}
		if (!slen) {
			perr("Sequence #%d is empty", num);
			return PARSER_ERROR;
		}
		if (!sequence_length_limit(slen)) {
			perr("Sequence #%d exceeds length limits", num);
			return PARSER_ERROR;
		}
		if (sum + slen + 1 > S32_MAX) {
			perr("Length overflow after %d sequences", num);
			return PARSER_ERROR;
		}
		max = max(max, slen);
		sum += slen + 1;
		*w++ = '\0';
	}
	in->max = max;
	in->num = num;
	return PARSER_SUCCESS;
}

SOURCE_REGISTER(fasta, parse_fasta);
