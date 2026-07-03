#include "io/reader.h"

#include <ctype.h>
#include <limits.h>
#include <print.h>
#include <stdlib.h>
#include <string.h>

#include "bio/align.h"
#include "system/os.h"
#include "util/benchmark.h"

static const struct dsv_pair {
	const char *ext;
	u8 delimiter;
} DSV_PAIRS[] = {
	{ "csv", ',' }, { "tsv", '\t' }, { "ssv", ';' }, { "psv", '|' }, {},
};

static const char *KEYS[] = {
	"sequence", "seq",     "protein", "dna",   "rna",
	"amino",    "peptide", "chain",	  nullptr,
};

static const u8 *dsv_field(const u8 **cur, const u8 *end, u8 delim, s32 *flen)
{
	const u8 *p = *cur;
	const u8 *start = p;
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

	s32 len = p - start;
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

static s32 dsv_cols(const u8 *p, const u8 *end, u8 delim)
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

static enum reader_result read_dsv(struct reader r, struct input *in)
{
	pverbm("Trying out DSV reader");
	const struct dsv_pair *pair = DSV_PAIRS;
	for (; pair->ext; pair++) {
		if (strcasecmp(pair->ext, r.ext) == 0)
			break;
	}
	if (!pair->ext)
		return READER_UNSUPPORTED;

	pverbm("Using DSV reader");
	const u8 *p = r.file;
	const u8 *header_line = p;
	u8 delim = pair->delimiter;
	s32 cols = dsv_cols(p, r.fend, delim);

	const char **MALLOCA(headers, cols + 1);
	if (!headers) {
		perr("Out of memory during DSV parsing");
		return READER_ERROR;
	}

	for (s32 col = 0; col < cols; col++) {
		s32 flen;
		const u8 *field = dsv_field(&p, r.fend, delim, &flen);
		if (!flen) {
			for (s32 j = 0; j < col; j++)
				free((void *)headers[j]);
			free(headers);
			perr("First row has empty column");
			return READER_ERROR;
		}
		char *MALLOCA(header, (flen + 1));
		if (!header) {
			for (s32 j = 0; j < col; j++)
				free((void *)headers[j]);
			free(headers);
			perr("Out of memory during DSV parsing");
			return READER_ERROR;
		}
		memcpy(header, field, flen);
		header[flen] = '\0';
		headers[col] = header;
	}
	while (p < r.fend && (*p == '\n' || *p == '\r'))
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
		free((void *)headers[col]);
	free(headers);

	s32 num = 0;
	s32 max = 0;
	s64 sum = 0;
	u8 *w = r.file;
	while (p < r.fend) {
		while (p < r.fend && (*p == '\n' || *p == '\r'))
			p++;
		if (p >= r.fend)
			break;

		num++;
		s32 flen = 0;
		for (s32 col = 0; col < seq_col; col++) {
			dsv_field(&p, r.fend, delim, &flen);
			if (p >= r.fend || *p == '\n' || *p == '\r') {
				perr("DSV row #%d has no sequence column", num);
				return READER_ERROR;
			}
		}
		const u8 *field = dsv_field(&p, r.fend, delim, &flen);
		if (!flen) {
			perr("Sequence #%d is empty", num);
			return READER_ERROR;
		}

		s32 slen = 0;
		for (s32 i = 0; i < flen; i++) {
			u8 c = toupper(field[i]);
			if (c == '\r' || c == '\n' || c == ' ' || c == '"')
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

		for (s32 i = seq_col + 1; i < cols; i++) {
			if (p >= r.fend || *p == '\n' || *p == '\r') {
				perr("DSV row #%d has too few columns", num);
				return READER_ERROR;
			}
			dsv_field(&p, r.fend, delim, &flen);
		}
		if (p < r.fend && *p != '\n' && *p != '\r') {
			perr("DSV row #%d has too many columns", num);
			return READER_ERROR;
		}
	}
	in->max = max;
	in->num = num;
	pverbl("DSV parsing finished successfuly");
	return READER_SUCCESS;
}

READER_REGISTER(dsv, read_dsv);
