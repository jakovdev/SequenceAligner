#include "io/input.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>

extern "C" {
#include <args.h>
#include <print.h>

#include "bio/alignment.h"
#include "bio/sequences.h"
#include "system/memory.h"
#include "system/os.h"
#include "util/benchmark.h"
}

std::string_view trim(std::string_view text) noexcept
{
	size_t start = 0, end = text.size();
	while (start < end && std::isspace((uchar)text[start]))
		start++;
	while (end > start && std::isspace((uchar)text[end - 1]))
		end--;
	return text.substr(start, end - start);
}

static bool sequence_normalize(std::string &seq) noexcept
{
	for (char &ch : seq) {
		ch = (char)std::toupper((uchar)ch);
		if (SEQ_LUT[(uchar)ch] < 0)
			return false;
	}
	return true;
}

static bool sequence_length_limit(size_t len) noexcept
{
	const s32 gap = -(GAP_PEN);
	if (!gap)
		return len <= SEQ_LEN_MAX;
	return len <= SEQ_LEN_MAX / (size_t)gap;
}

source::source(const char *path) noexcept
{
	std::string_view path_view(path);
	size_t dot = path_view.rfind('.');
	if (dot == std::string::npos) {
		perr("File extension not found: %s", file_name(path));
		return;
	}
	this->extension = path_view.substr(dot + 1);
	for (char &ch : this->extension)
		ch = (char)std::tolower((uchar)ch);

	std::ifstream f(path, std::ios::binary);
	if (!f) {
		perr("Could not open file: %s", file_name(path));
		return;
	}

	std::string data;
	data.assign(std::istreambuf_iterator<char>(f),
		    std::istreambuf_iterator<char>());

	if (f.bad()) {
		perr("Failed to read file: %s", file_name(path));
		return;
	}

	if (data.find('\0') != std::string::npos) {
		perr("File contains null bytes which may indicate corruption: %s",
		     file_name(path));
		return;
	}

	this->lines.reserve(1 + std::ranges::count(data, '\n'));
	for (size_t i = 0; i < data.size();) {
		size_t end = i;
		while (end < data.size() && data[end] != '\n' &&
		       data[end] != '\r')
			end++;

		auto line = trim(std::string_view(data.data() + i, end - i));
		if (!line.empty())
			this->lines.emplace_back(line);

		if (end < data.size() && data[end] == '\r' &&
		    end + 1 < data.size() && data[end + 1] == '\n')
			i = end + 2;
		else
			i = end + 1;
	}

	for (auto parser : parsers) {
		if (parser(*this))
			break;
	}
}

bool source::load(struct sequences *S) noexcept
{
	if (lines.empty())
		return false;

	if (seqs.empty()) {
		perr("Unsupported file format");
		return false;
	}

	if (seqs.size() < SEQ_N_MIN) {
		perr("Not enough sequences: %zu (min: %d)", seqs.size(),
		     SEQ_N_MIN);
		return false;
	}
	if (seqs.size() > SEQ_N_MAX) {
		perr("Too many sequences: %zu (max: %d)", seqs.size(),
		     SEQ_N_MAX);
		return false;
	}

	size_t total = 0;
	for (const auto &seq : seqs)
		total += seq.size() + 1;

	sequences_free(S);
	MALLOCA_AL(S->seqs, CACHE_LINE, seqs.size());
	MALLOCA_AL(S->lengths, CACHE_LINE, seqs.size());
	MALLOCA_AL(S->offsets, CACHE_LINE, seqs.size());
	MALLOCA_AL(S->letters, PAGE_SIZE, total);
	if (!S->lengths || !S->offsets || !S->seqs || !S->letters) {
		perr("Out of memory allocating %zu sequences", seqs.size());
		return false;
	}

	int large = -1;
	s32 seq_n_long = 0;
	int invalid = -1;
	s32 seq_n_invalid = 0;
	for (size_t i = 0, offset = 0; i < seqs.size(); i++) {
		auto &seq = seqs[i];
		if (!sequence_length_limit(seq.size())) {
			if (large < 0) {
				bench_io_end();
				pwarn("Sequence %zu exceeds length limits",
				      i + 1);
				large = print_yN("Skip long sequences?");
				bench_io_start();
			}

			if (large > 0) {
				seq_n_long++;
				continue;
			}

			perr("Sequence %zu exceeds length limits", i + 1);
			return false;
		}

		if (!sequence_normalize(seq)) {
			if (invalid < 0) {
				bench_io_end();
				pwarn("Sequence %zu has invalid letters",
				      i + 1);
				invalid = print_yN("Skip invalid sequences?");
				bench_io_start();
			}

			if (invalid > 0) {
				seq_n_invalid++;
				continue;
			}

			perr("Sequence %zu has invalid letters", i + 1);
			return false;
		}

		char *dst = S->letters + offset;
		std::memcpy(dst, seq.data(), seq.size());
		dst[seq.size()] = '\0';

		S->lengths[S->seqs_n] = (s32)seq.size();
		S->offsets[S->seqs_n] = (s64)offset;
		S->seqs[S->seqs_n].letters = dst;
		S->seqs[S->seqs_n].length = (s32)seq.size();
		S->lengths_max = std::max(S->lengths_max, seq.size());

		offset += seq.size() + 1;
		S->seqs_n++;
	}

	if (seq_n_long)
		pinfo("Skipped %w32d sequences that were too long", seq_n_long);

	if (seq_n_invalid)
		pinfo("Skipped %w32d invalid sequences", seq_n_invalid);

	if (S->seqs_n < SEQ_N_MIN) {
		perr("Not enough valid sequences: %w32d (min: %d)", S->seqs_n,
		     SEQ_N_MIN);
		return false;
	}

	S->alignments = ((s64)S->seqs_n * (S->seqs_n - 1)) / 2;
	s64 t_len = S->offsets[S->seqs_n - 1] + S->lengths[S->seqs_n - 1] + 1;
	S->average_length = (double)t_len / (double)S->seqs_n - 1.0;
	return true;
}

extern "C" {

static const char *INPUT_PATH;

void sequences_free(struct sequences *S)
{
	free_aligned(S->lengths);
	free_aligned(S->offsets);
	free_aligned(S->letters);
	free_aligned(S->seqs);
	std::memset(S, 0, sizeof(*S));
}

bool sequences_load(struct sequences *S)
{
	bench_io_start();

	source src(INPUT_PATH);
	if (!src.load(S))
		return false;

	bench_io_end();
	return true;
}

bool sequences_lose(struct sequences *S, const bool *lost)
{
	s64 used = 0;
	s32 write = 0;
	S->lengths_max = 0;
	for (s32 read = 0; read < S->seqs_n; read++) {
		if (lost[read])
			continue;

		s32 len = S->lengths[read];
		s64 off = S->offsets[read];
		char *dst = S->letters + used;
		size_t LEN = (size_t)len;
		if (used != off)
			memmove(dst, S->letters + off, LEN + 1);
		S->lengths[write] = len;
		S->offsets[write] = used;
		S->seqs[write].length = len;
		S->seqs[write++].letters = dst;
		S->lengths_max = std::max(S->lengths_max, LEN);
		used += len + 1;
	}

	if (write < SEQ_N_MIN) {
		perr("Not enough filtered sequences: %w32d (min: %d)", write,
		     SEQ_N_MIN);
		return false;
	}

	S->seqs_n = write;
	S->alignments = ((s64)write * (write - 1)) / 2;
	s64 total = S->offsets[write - 1] + S->lengths[write - 1] + 1;
	S->average_length = (double)total / (double)write - 1.0;
	return true;
}

static void print_input_path(void)
{
	pinfo("Input: %s", file_name(INPUT_PATH));
}

static struct arg_callback validate_input_path(void)
{
	return !path_file_exists(INPUT_PATH) ? ARG_INVALID("File not found") :
					       ARG_VALID();
}

ARGUMENT(input_path) = {
	.set = {},
	.dest = &INPUT_PATH,
	.parse_callback = parse_path,
	.validate_callback = validate_input_path,
	.action_callback = print_input_path,
	.arg_req = ARG_REQUIRED,
	.param_req = ARG_PARAM_REQUIRED,
	.validate_phase = ARG_CALLBACK_IF_SET,
	.action_phase = ARG_CALLBACK_ALWAYS,
	.validate_order = ARG_ORDER_FIRST,
	.action_order = ARG_ORDER_FIRST,
	.help_order = ARG_ORDER_FIRST,
	.help = "Input file path: FASTA, DSV (CSV, TSV, etc.) format",
	.param = "FILE",
	.lopt = "input",
	.opt = 'i',
	._ = {}
};
}
