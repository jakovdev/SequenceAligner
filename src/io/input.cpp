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
#include "bio/sequence.h"
#include "io/input.h"
#include "system/os.h"
#include "util/benchmark.h"

bool filter(struct input *);
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
	s32 gap = -(GAP_PEN);
	if (!gap)
		return len <= SEQ_LEN_MAX;
	return len <= SEQ_LEN_MAX / (size_t)gap;
}

bool source::load(struct input *in, const char *path) noexcept
{
	std::string_view path_view(path);
	size_t dot = path_view.rfind('.');
	if (dot == std::string::npos) {
		perr("File extension not found: %s", file_name(path));
		return false;
	}
	this->extension = path_view.substr(dot + 1);
	for (char &ch : this->extension)
		ch = (char)std::tolower((uchar)ch);

	std::ifstream f(path, std::ios::binary);
	if (!f) {
		perr("Could not open file: %s", file_name(path));
		return false;
	}

	std::string data;
	data.assign(std::istreambuf_iterator<char>(f),
		    std::istreambuf_iterator<char>());

	if (f.bad()) {
		perr("Failed to read file: %s", file_name(path));
		return false;
	}

	if (data.find('\0') != std::string::npos) {
		perr("File contains null bytes which may indicate corruption: %s",
		     file_name(path));
		return false;
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

	if (this->lines.empty()) {
		perr("Empty file: %s", file_name(path));
		return false;
	}

	for (auto parse : this->parsers) {
		switch (parse(*this)) {
		case parse_result::SUCCESS:
			goto parse_success;
		case parse_result::ERROR:
			return false;
		case parse_result::UNSUPPORTED:
			continue;
		}
	}

	perr("Unsupported file format: %s", file_name(path));
	return false;
parse_success:
	size_t seq_n = this->seqs.size();
	if (seq_n < SEQ_N_MIN) {
		perr("Not enough sequences: %zu (min: %d)", seq_n, SEQ_N_MIN);
		return false;
	}
	if (seq_n > SEQ_N_MAX) {
		perr("Too many sequences: %zu (max: %d)", seq_n, SEQ_N_MAX);
		return false;
	}

	size_t total = 0;
	for (const auto &seq : this->seqs)
		total += seq.size() + 1;

	input_free(in);
	MALLOCA_AL(in->seqs, CACHE_LINE, seq_n);
	MALLOCA_AL(in->lengths, CACHE_LINE, seq_n);
	MALLOCA_AL(in->offsets, CACHE_LINE, seq_n);
	MALLOCA_AL(in->letters, PAGE_SIZE, total);
	if (!in->lengths || !in->offsets || !in->seqs || !in->letters) {
		perr("Out of memory for %zu sequences", seq_n);
		return false;
	}

	int large = -1;
	s32 seq_n_long = 0;
	int invalid = -1;
	s32 seq_n_invalid = 0;
	s32 i = 0;
	s64 offset = 0;
	for (auto &seq : this->seqs) {
		i++;
		if (!sequence_length_limit(seq.size())) {
			if (large < 0) {
				bench_input_end();
				pwarn("Sequence #%d exceeds length limits", i);
				large = print_yN("Skip long sequences?");
				bench_input_start();
			}

			if (large > 0) {
				seq_n_long++;
				continue;
			}

			perr("Sequence #%d exceeds length limits", i);
			return false;
		}

		if (!sequence_normalize(seq)) {
			if (invalid < 0) {
				bench_input_end();
				pwarn("Sequence #%d has invalid letters", i);
				invalid = print_yN("Skip invalid sequences?");
				bench_input_start();
			}

			if (invalid > 0) {
				seq_n_invalid++;
				continue;
			}

			perr("Sequence #%d has invalid letters", i);
			return false;
		}

		char *dst = in->letters + offset;
		std::memcpy(dst, seq.data(), seq.size());
		dst[seq.size()] = '\0';
		s32 len = (s32)seq.size();

		in->lengths[in->seqs_n] = len;
		in->offsets[in->seqs_n] = offset;
		in->seqs[in->seqs_n].letters = dst;
		in->seqs[in->seqs_n++].length = len;
		in->lengths_max = std::max(in->lengths_max, len);
		offset += len + 1;
	}

	if (seq_n_long)
		pinfo("Skipped %d sequences that were too long", seq_n_long);

	if (seq_n_invalid)
		pinfo("Skipped %d invalid sequences", seq_n_invalid);

	if (in->seqs_n < SEQ_N_MIN) {
		perr("Not enough valid sequences: %d (min: %d)", in->seqs_n,
		     SEQ_N_MIN);
		return false;
	}

	return filter(in);
}

extern "C" {

static const char *INPUT_PATH;

void input_free(struct input *in)
{
	free_aligned(in->lengths);
	free_aligned(in->offsets);
	free_aligned(in->letters);
	free_aligned(in->seqs);
	std::memset(in, 0, sizeof(*in));
}

bool input_load(struct input *in)
{
	bench_input_start();

	source src{};
	if (!src.load(in, INPUT_PATH))
		return false;

	bench_input_end();
	in->alignments = ((s64)in->seqs_n * (in->seqs_n - 1)) / 2;
	s64 sum = in->offsets[in->seqs_n - 1] + in->lengths[in->seqs_n - 1] + 1;
	double average_length = (double)sum / (double)in->seqs_n - 1.0;
	pinfo("Loaded %d sequences", in->seqs_n);
	pinfo("Average sequence length: %.2f", average_length);
	bench_input_print();
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
