#include "io/source.h"

#include <ctype.h>
#include <limits.h>
#include <print.h>
#include <stdlib.h>
#include <string.h>

#include "bio/align.h"
#include "io/input.h"
#include "system/os.h"
#include "util/benchmark.h"

static const struct dsv_pair {
	const char *ext;
	uchar delimiter;
} DSV_PAIRS[] = {
	{ "csv", ',' }, { "tsv", '\t' }, { "ssv", ';' }, { "psv", '|' }, {},
};

static const char *KEYS[] = {
	"sequence", "seq",     "protein", "dna",   "rna",
	"amino",    "peptide", "chain",	  nullptr,
};

static const uchar *dsv_field(const uchar **cur, const uchar *end, uchar delim,
			      s32 *flen)
{
	const uchar *p = *cur;
	const uchar *start = p;
	bool quoted = false;
	while (p < end) {
		if (*p == '"') {
			if (quoted && p + 1 < end && p[1] == '"') {
				p += 2;
				continue;
			}
			quoted = !quoted;
			p++;
			continue;
		}
		if (!quoted && (*p == delim || *p == '\n' || *p == '\r'))
			break;
		p++;
	}

	s32 len = (s32)(p - start);
	if (len >= 2 && *start == '"' && start[len - 1] == '"') {
		len -= 2;
		start++;
	}
	*flen = len;
	if (p < end && *p == delim)
		p++;
	*cur = p;
	return start;
}

static s32 dsv_cols(const uchar *p, const uchar *end, uchar delim)
{
	s32 count = 1;
	bool quoted = false;
	while (p < end) {
		if (*p == '"') {
			if (quoted && p + 1 < end && p[1] == '"') {
				p += 2;
				continue;
			}
			quoted = !quoted;
		} else if (*p == delim && !quoted) {
			count++;
		}
		if (!quoted && (*p == '\n' || *p == '\r'))
			break;
		p++;
	}
	return count;
}

static enum parse_result parse_dsv(struct source src, struct input *in)
{
	const struct dsv_pair *pair = DSV_PAIRS;
	for (; pair->ext; pair++) {
		if (strcasecmp(pair->ext, src.ext) == 0)
			break;
	}
	if (!pair->ext)
		return PARSER_UNSUPPORTED;

	const uchar *p = src.file;
	const uchar *header_line = p;
	uchar delim = pair->delimiter;
	s32 cols = dsv_cols(p, src.fend, delim);

	const char **MALLOCA(headers, cols + 1);
	if (!headers) {
		perr("Out of memory during DSV parsing");
		return PARSER_ERROR;
	}

	for (s32 col = 0; col < cols; col++) {
		s32 flen;
		const uchar *field = dsv_field(&p, src.fend, delim, &flen);
		if (!flen) {
			for (s32 j = 0; j < col; j++)
				free((char *)headers[j]);
			free(headers);
			perr("First row has empty column");
			return PARSER_ERROR;
		}
		char *MALLOCA(header, flen + 1);
		if (!header) {
			for (s32 j = 0; j < col; j++)
				free((char *)headers[j]);
			free(headers);
			perr("Out of memory during DSV parsing");
			return PARSER_ERROR;
		}
		memcpy(header, field, flen);
		header[flen] = '\0';
		headers[col] = header;
	}
	while (p < src.fend && (*p == '\n' || *p == '\r'))
		p++;

	s32 seq_col = -1;
	for (s32 col = 0; col < cols && seq_col < 0; col++) {
		for (const char **key = KEYS; *key; key++) {
			if (strcasecmp(headers[col], *key) == 0) {
				seq_col = col;
				break;
			}
		}
	}

	if (seq_col < 0) {
		bench_input_end();
		headers[cols] = "No header line";
		pinfol("Which column contains your sequences?");
		s32 choice = pchoice(headers, cols + 1, "Enter column number");
		if (choice == cols) {
			p = header_line;
			pinfol("Which column contains a sequence?");
			seq_col = pchoice(headers, cols, "Enter column number");
		} else {
			seq_col = choice;
		}
		bench_input_start();
	}

	for (s32 col = 0; col < cols; col++)
		free((char *)headers[col]);
	free(headers);

	s32 num = 0;
	s32 max = 0;
	s64 sum = 0;
	uchar *w = src.file;
	while (p < src.fend) {
		while (p < src.fend && (*p == '\n' || *p == '\r'))
			p++;
		if (p >= src.fend)
			break;

		num++;
		s32 flen = 0;
		for (s32 col = 0; col < seq_col; col++) {
			dsv_field(&p, src.fend, delim, &flen);
			if (p >= src.fend || *p == '\n' || *p == '\r') {
				perr("DSV row #%d has no sequence column", num);
				return PARSER_ERROR;
			}
		}
		const uchar *field = dsv_field(&p, src.fend, delim, &flen);
		if (!flen) {
			perr("Sequence #%d is empty", num);
			return PARSER_ERROR;
		}

		s32 slen = 0;
		for (s32 i = 0; i < flen; i++) {
			uchar c = (uchar)toupper(field[i]);
			if (c == '\r' || c == '\n' || c == ' ' || c == '"')
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

		for (s32 i = seq_col + 1; i < cols; i++) {
			if (p >= src.fend || *p == '\n' || *p == '\r') {
				perr("DSV row #%d has too few columns", num);
				return PARSER_ERROR;
			}
			dsv_field(&p, src.fend, delim, &flen);
		}
		if (p < src.fend && *p != '\n' && *p != '\r') {
			perr("DSV row #%d has too many columns", num);
			return PARSER_ERROR;
		}
	}
	in->max = max;
	in->num = num;
	return PARSER_SUCCESS;
}

SOURCE_REGISTER(dsv, parse_dsv);
