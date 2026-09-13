#include "io/reader.h"

#include <ctype.h>
#include <limits.h>
#include <print.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "bio/method.h"
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

static const u8 *dsv_field(const u8 **cur, const u8 *end, u8 delim, uhz *flen)
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

	uhz len = p - start;
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

static uhz dsv_cols(const u8 *p, const u8 *end, u8 delim)
{
	uhz count = 1;
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

	pverbl("Using DSV reader");
	const u8 *p = r.file;
	const u8 *header_line = p;
	u8 delim = pair->delimiter;
	uhz cols = dsv_cols(p, r.fend, delim);

	const char **MALLOCA(headers, cols + 1);
	if (!headers) {
		perr("Out of memory during DSV parsing");
		return READER_ERROR;
	}

	u8 *hw;
	for (uhz col = 0; col < cols; col++) {
		uhz flen;
		hw = (u8 *)dsv_field(&p, r.fend, delim, &flen);
		if (!flen) {
			free(headers);
			perr("First row has empty column");
			return READER_ERROR;
		}
		hw[flen] = '\0';
		headers[col] = (const char *)hw;
	}
	while (p < r.fend && (*p == '\n' || *p == '\r'))
		p++;

	uhz seq_col = UHZ_MAX;
	for (uhz col = 0; col < cols && seq_col != UHZ_MAX; col++) {
		for (const char **key = KEYS; *key; key++) {
			if (strcasecmp(headers[col], *key) == 0) {
				seq_col = col;
				break;
			}
		}
	}

	if (seq_col == UHZ_MAX) {
		bench_input_end();
		headers[cols] = "No header line! Do not skip!";
		pinfo("Under which header are sequences? Header is skipped!");
		uhz choice = pchoice(headers, cols + 1, "Enter range");
		if (choice == cols) {
			p = header_line;
			pinfol("Which DSV column displays a sequence?");
			seq_col = pchoice(headers, cols, "Enter range");
		} else {
			seq_col = choice;
		}
		bench_input_start();
	}

	for (uhz i = 0; i < cols - 1; i++) {
		hw = (u8 *)headers[i];
		hw[strlen(headers[i])] = delim;
	}
	hw = (u8 *)headers[cols - 1];
	hw[strlen(headers[cols - 1])] = '\n';
	free(headers);

	uhz num = 0;
	uhz max = 0;
	usz sum = 0;
	u8 *w = r.file;
	while (p < r.fend) {
		while (p < r.fend && (*p == '\n' || *p == '\r'))
			p++;
		if (p >= r.fend)
			break;

		num++;
		uhz flen = 0;
		for (uhz col = 0; col < seq_col; col++) {
			dsv_field(&p, r.fend, delim, &flen);
			if (p >= r.fend || *p == '\n' || *p == '\r') {
				perr("DSV row #%u has no sequence column", num);
				return READER_ERROR;
			}
		}
		const u8 *field = dsv_field(&p, r.fend, delim, &flen);
		if (!flen) {
			perr("Sequence #%u is empty", num);
			return READER_ERROR;
		}

		uhz slen = 0;
		for (uhz i = 0; i < flen; i++) {
			u8 c = toupper(field[i]);
			if (c == '\r' || c == '\n' || c == ' ' || c == '"')
				continue;
			if (c == '\0' || c > SCHAR_MAX) {
				perr("Sequence #%u is corrupted", num);
				return READER_ERROR;
			}
			if (SEQ_LUT[c] < 0) {
				perr("Sequence #%u is invalid", num);
				return READER_ERROR;
			}
			*w++ = c;
			slen++;
		}
		if (!slen) {
			perr("Sequence #%u is empty", num);
			return READER_ERROR;
		}
		if (!sequence_length_limit(slen)) {
			perr("Sequence #%u exceeds length limits", num);
			return READER_ERROR;
		}
		if (sum + slen + 1 > UHZ_MAX) {
			perr("Length overflow after %u sequences", num);
			return READER_ERROR;
		}
		max = max(max, slen);
		sum += slen + 1;
		*w++ = '\0';

		for (uhz i = seq_col + 1; i < cols; i++) {
			if (p >= r.fend || *p == '\n' || *p == '\r') {
				perr("DSV row #%u has too few columns", num);
				return READER_ERROR;
			}
			dsv_field(&p, r.fend, delim, &flen);
		}
		if (p < r.fend && *p != '\n' && *p != '\r') {
			perr("DSV row #%u has too many columns", num);
			return READER_ERROR;
		}
	}
	in->max = max;
	in->num = num;
	pverb("DSV parsing finished successfuly");
	return READER_SUCCESS;
}

READER_REGISTER(dsv, read_dsv);
