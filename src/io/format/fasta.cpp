#include "io/input.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>

extern "C" {
#include <print.h>

#include "system/types.h"
}

using result = source::parse_result;

constexpr std::array EXTS = { "fasta", "fa",  "fas", "fna",
			      "ffn",   "faa", "frn", "mpfa" };

static result parse_fasta(source &src) noexcept
{
	if (std::ranges::find(EXTS, src.extension) == EXTS.end())
		return result::UNSUPPORTED;

	if (std::ranges::none_of(src.lines, [](auto line) {
		    return !line.empty() && line.starts_with('>');
	    })) {
		return result::UNSUPPORTED;
	}

	std::string sequence{};
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

	size_t n{};
	for (auto line : src.lines) {
		n++;
		if (line.front() == '>') {
			if (!flush_sequence()) {
				perr("Empty sequence at line %zu", n);
				return result::ERROR;
			}
			in_seq = true;
			continue;
		}

		if (!in_seq) {
			perr("Data before first header");
			return result::ERROR;
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
		return result::ERROR;
	}

	if (!flush_sequence()) {
		perr("Last header has no data");
		return result::ERROR;
	}

	return result::SUCCESS;
}

static input_format fasta_format{ parse_fasta };
