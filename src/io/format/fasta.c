#include "io/source.h"

#include <print.h>
#include <string.h>

#include "system/os.h"

static const char *EXTS[] = {
	"fasta", "fa", "fas", "fna", "ffn", "faa", "frn", "mpfa", nullptr,
};

static enum source_result parse_fasta(struct source *src)
{
	const char **ext = EXTS;
	for (; *ext; ext++) {
		if (strcasecmp(*ext, src->ext) == 0)
			break;
	}
	if (!*ext)
		return SOURCE_UNSUPPORTED;

	const uchar *p = src->file;
	const uchar *end = src->fend;
	if (*p != '>') {
		perr("Data before first header");
		return SOURCE_ERROR;
	}

	s32 num = 0;
	s32 sum = 0;
	const uchar *cur = p;
	while (cur < end) {
		while (cur < end && *cur != '\n' && *cur != '\r')
			cur++;
		while (cur < end && (*cur == '\n' || *cur == '\r'))
			cur++;

		if (cur >= end) {
			perr("Last header has no data");
			return SOURCE_ERROR;
		}

		s32 seq_len = 0;
		while (cur < end && *cur != '>') {
			const uchar *ls = cur;
			while (cur < end && *cur != '\n' && *cur != '\r')
				cur++;
			s32 ll = (s32)(cur - ls);
			if (ll > 0) {
				if ((s64)sum + ll > S32_MAX) {
					perr("Too many large sequences");
					return SOURCE_ERROR;
				}
				seq_len += ll;
				sum += ll;
			}
			while (cur < end && (*cur == '\n' || *cur == '\r'))
				cur++;
		}
		num++;
		if (!seq_len) {
			perr("Sequence #%d is empty", num);
			return SOURCE_ERROR;
		}
	}

	if (!num) {
		perr("No sequences found");
		return SOURCE_ERROR;
	}

	struct entry *MALLOCA(entries, num);
	if (!entries) {
		perr("Out of memory during FASTA parsing");
		return SOURCE_ERROR;
	}

	cur = p;
	for (s32 i = 0; i < num; i++) {
		while (cur < end && *cur != '\n' && *cur != '\r')
			cur++;
		while (cur < end && (*cur == '\n' || *cur == '\r'))
			cur++;

		const uchar *seq_start = nullptr;
		const uchar *seq_end = nullptr;
		while (cur < end && *cur != '>') {
			const uchar *ls = cur;
			while (cur < end && *cur != '\n' && *cur != '\r')
				cur++;
			s32 ll = (s32)(cur - ls);
			if (ll > 0) {
				if (!seq_start)
					seq_start = ls;
				seq_end = cur;
			}
			while (cur < end && (*cur == '\n' || *cur == '\r'))
				cur++;
		}

		entries[i].off = (s32)(seq_start - src->file);
		entries[i].len = (s32)(seq_end - seq_start);
	}

	src->entries = entries;
	src->num = num;
	src->sum = sum;
	return SOURCE_SUCCESS;
}

SOURCE_REGISTER(fasta, parse_fasta);
