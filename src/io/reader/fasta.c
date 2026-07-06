#include "io/reader.h"

#include <ctype.h>
#include <limits.h>
#include <print.h>
#include <strings.h>

#include "bio/method.h"
#include "util/macros.h"

static const char *EXTS[] = {
	"fasta", "fa", "fas", "fna", "ffn", "faa", "frn", "mpfa", nullptr,
};

static enum reader_result read_fasta(struct reader r, struct input *in)
{
	pverbm("Trying out FASTA reader");
	const char **ext = EXTS;
	for (; *ext; ext++) {
		if (strcasecmp(*ext, r.ext) == 0)
			break;
	}
	if (!*ext)
		return READER_UNSUPPORTED;

	pverbl("Using FASTA reader");
	const u8 *p = r.file;
	if (*p != '>') {
		perr("Data before first header");
		return READER_ERROR;
	}

	s32 num = 0;
	s32 max = 0;
	s64 sum = 0;
	u8 *w = r.file;
	while (p < r.fend) {
		while (p < r.fend && *p != '\n' && *p != '\r')
			p++;
		while (p < r.fend && (*p == '\n' || *p == '\r'))
			p++;
		if (p >= r.fend) {
			perr("Last header has no data");
			return READER_ERROR;
		}

		num++;
		s32 slen = 0;
		while (p < r.fend && *p != '>') {
			u8 c = toupper(*p++);
			if (c == '\r' || c == '\n' || c == ' ')
				continue;
			if (c == '\0' || c > SCHAR_MAX) {
				perr("Sequence #%d is corrupted", num);
				return READER_ERROR;
			}
			if (SEQ_LUT[c] < 0) {
				perr("Sequence #%d is invalid", num);
				return READER_ERROR;
			}
			*w++ = c;
			slen++;
		}
		if (!slen) {
			perr("Sequence #%d is empty", num);
			return READER_ERROR;
		}
		if (!sequence_length_limit(slen)) {
			perr("Sequence #%d exceeds length limits", num);
			return READER_ERROR;
		}
		if (sum + slen + 1 > S32_MAX) {
			perr("Length overflow after %d sequences", num);
			return READER_ERROR;
		}
		max = max(max, slen);
		sum += slen + 1;
		*w++ = '\0';
	}
	in->max = max;
	in->num = num;
	pverb("FASTA parsing finished successfuly");
	return READER_SUCCESS;
}

READER_REGISTER(fasta, read_fasta);
