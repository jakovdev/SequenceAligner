#include "io/source.h"

#include <print.h>
#include <stdlib.h>
#include <string.h>

#include "system/os.h"
#include "system/types.h"
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
	const uchar *field_start = p;
	bool quoted = false;
	while (p < end && *p != '\n' && *p != '\r') {
		if (*p == '"') {
			if (quoted && p + 1 < end && *(p + 1) == '"') {
				p += 2;
				continue;
			}
			quoted = !quoted;
			p++;
			continue;
		}
		if (*p == delim && !quoted)
			break;
		p++;
	}

	*flen = (s32)(p - field_start);
	if (p < end && *p == delim)
		p++;
	*cur = p;
	return field_start;
}

static s32 dsv_cols(const uchar *p, const uchar *end, uchar delim)
{
	s32 count = 1;
	bool quoted = false;
	while (p < end && *p != '\n' && *p != '\r') {
		if (*p == '"') {
			if (quoted && p + 1 < end && *(p + 1) == '"') {
				p += 2;
				continue;
			}
			quoted = !quoted;
		} else if (*p == delim && !quoted) {
			count++;
		}
		p++;
	}
	return count;
}

static enum source_result parse_dsv(struct source *src)
{
	const struct dsv_pair *pair = DSV_PAIRS;
	for (; pair->ext; pair++) {
		if (strcasecmp(pair->ext, src->ext) == 0)
			break;
	}
	if (!pair->ext)
		return SOURCE_UNSUPPORTED;

	const uchar *p = src->file;
	const uchar *end = src->fend;
	uchar delim = pair->delimiter;
	s32 cols = dsv_cols(p, end, delim);
	if (!cols) {
		perr("No sequences found");
		return SOURCE_ERROR;
	}

	const char **MALLOCA(headers, cols + 2);
	if (!headers) {
		perr("Out of memory during DSV parsing");
		return SOURCE_ERROR;
	}

	const uchar *header_line = p;
	for (s32 col = 0; col < cols; col++) {
		s32 flen;
		const uchar *field = dsv_field(&p, end, delim, &flen);
		if (flen >= 2 && field[0] == '"' && field[flen - 1] == '"') {
			field++;
			flen -= 2;
		}
		while (flen > 0 && field[0] == ' ') {
			field++;
			flen--;
		}
		while (flen > 0 && field[flen - 1] == ' ')
			flen--;

		char *MALLOCA(header, flen + 1);
		if (!header) {
			for (s32 j = 0; j < col; j++)
				free((char *)headers[j]);
			free(headers);
			perr("Out of memory during DSV parsing");
			return SOURCE_ERROR;
		}
		memcpy(header, field, flen);
		header[flen] = '\0';
		headers[col] = header;
	}
	while (p < end && *p != '\n' && *p != '\r')
		p++;
	while (p < end && (*p == '\n' || *p == '\r'))
		p++;

	s32 seq_col = -1;
	s32 first_row = 1;
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
			first_row = 0;
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
	s32 sum = 0;
	s32 row = first_row;
	const uchar *scan = p;
	while (scan < end) {
		while (scan < end && (*scan == '\n' || *scan == '\r'))
			scan++;
		if (scan >= end)
			break;

		if (cols != dsv_cols(scan, end, delim)) {
			perr("Column count mismatch at row %d", row);
			return SOURCE_ERROR;
		}

		const uchar *lp = scan;
		s32 flen = 0;
		for (s32 col = 0; col <= seq_col; col++)
			dsv_field(&lp, end, delim, &flen);

		if (!flen) {
			perr("Empty sequence at row %d", row);
			return SOURCE_ERROR;
		}
		if ((s64)sum + flen > S32_MAX) {
			perr("Too many large sequences");
			return SOURCE_ERROR;
		}

		sum += flen;
		num++;
		row++;

		while (scan < end && *scan != '\n' && *scan != '\r')
			scan++;
		while (scan < end && (*scan == '\n' || *scan == '\r'))
			scan++;
	}

	if (!num) {
		perr("No sequences found");
		return SOURCE_ERROR;
	}

	struct entry *MALLOCA(entries, num);
	if (!entries) {
		perr("Out of memory during DSV parsing");
		return SOURCE_ERROR;
	}

	scan = p;
	for (s32 i = 0; i < num; i++) {
		while (scan < end && (*scan == '\n' || *scan == '\r'))
			scan++;

		const uchar *lp = scan;
		s32 flen = 0;
		const uchar *field = nullptr;
		for (s32 col = 0; col <= seq_col; col++)
			field = dsv_field(&lp, end, delim, &flen);

		entries[i].off = (s32)(field - src->file);
		entries[i].len = flen;

		while (scan < end && *scan != '\n' && *scan != '\r')
			scan++;
		while (scan < end && (*scan == '\n' || *scan == '\r'))
			scan++;
	}

	src->entries = entries;
	src->num = num;
	src->sum = sum;
	return SOURCE_SUCCESS;
}

SOURCE_REGISTER(dsv, parse_dsv);