#include "io/input.hpp"

#include <array>
#include <cctype>
#include <string>
#include <string_view>

#include "system/types.h"

extern "C" {
#include <print.h>
}

constexpr std::array EXTS = { "fasta", "fa",  "fas", "fna",
			      "ffn",   "faa", "frn", "mpfa" };

static bool parse_fasta(source &src) noexcept
{
	if (std::ranges::find(EXTS, src.extension) == EXTS.end())
		return false;

	bool fasta{};
	for (auto line : src.lines) {
		if (line.front() == '>') {
			fasta = true;
			break;
		}
	}
	if (!fasta)
		return false;

	std::string sequence;
	bool in_seq{};
	src.seqs.reserve(src.lines.size());
	auto flush_sequence = [&]() -> bool {
		if (!in_seq)
			return true;
		if (sequence.empty())
			return false;
		src.seqs.push_back(std::move(sequence));
		sequence.clear();
		return true;
	};

	size_t n = 0;
	for (auto line : src.lines) {
		n++;
		if (line.front() == '>') {
			if (!flush_sequence()) {
				perr("Empty sequence at line %zu", n);
				src.seqs.clear();
				return false;
			}
			in_seq = true;
			continue;
		}

		if (!in_seq) {
			perr("Data before first header");
			return false;
		}

		if (line.find_first_of(" \t\v\f\r") == std::string_view::npos) {
			sequence.append(line.data(), line.size());
			continue;
		}

		sequence.reserve(sequence.size() + line.size());
		for (uchar ch : line) {
			if (!std::isspace(ch))
				sequence.push_back((char)ch);
		}
	}

	if (!in_seq) {
		perr("No sequences found");
		return false;
	}

	if (!flush_sequence()) {
		perr("Last header has no data");
		src.seqs.clear();
		return false;
	}

	return true;
}

static format fasta_format(parse_fasta);
