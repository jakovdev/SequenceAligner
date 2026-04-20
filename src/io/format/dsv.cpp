#include "io/input.hpp"

#include <array>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

#include "system/types.h"

extern "C" {
#include <print.h>

#include "system/memory.h"
#include "util/benchmark.h"
}

struct dsv_pair {
	std::string_view extension{};
	char delimiter{};
};

constexpr std::array DSV_PAIRS = {
	dsv_pair{ "csv", ',' },
	dsv_pair{ "tsv", '\t' },
	dsv_pair{ "ssv", ';' },
	dsv_pair{ "psv", '|' },
};

constexpr std::array KEYS = {
	"sequence", "seq", "protein", "dna", "rna", "amino", "peptide", "chain",
};

static void dsv_split(std::vector<std::string_view> &tokens,
		      std::string_view line, char delimiter) noexcept
{
	tokens.clear();
	size_t token_start = 0;
	bool quoted = false;
	for (size_t i = 0; i < line.size(); i++) {
		char ch = line[i];
		if (ch == '"') {
			size_t next = i + 1;
			if (quoted && next < line.size() && line[next] == '"') {
				i++;
				continue;
			}
			if (!quoted && i == token_start)
				token_start = i + 1;
			quoted = !quoted;
			continue;
		}
		if (ch == delimiter && !quoted) {
			tokens.emplace_back(trim(
				line.substr(token_start, i - token_start)));
			token_start = i + 1;
			continue;
		}
	}

	tokens.emplace_back(trim(line.substr(token_start)));
}

static bool parse_dsv(source &src) noexcept
{
	char delimiter{};
	for (const auto &pair : DSV_PAIRS) {
		if (pair.extension == src.extension) {
			delimiter = pair.delimiter;
			break;
		}
	}
	if (!delimiter)
		return false;

	if (src.lines.empty()) {
		perr("No sequences found in DSV file");
		return false;
	}

	std::vector<std::string_view> tokens{};
	dsv_split(tokens, src.lines.front(), delimiter);
	if (!tokens.size()) {
		perr("No sequences found in DSV file");
		return false;
	}

	std::vector<std::string> headers(tokens.begin(), tokens.end());

	int header{};
	size_t first_row{};
	for (size_t col = 0; col < headers.size() && !first_row; col++) {
		std::string column(headers[col]);
		for (char &ch : column)
			ch = (char)std::tolower((uchar)ch);
		for (auto key : KEYS) {
			if (column.find(key) != std::string::npos) {
				header = col;
				first_row = 1;
				break;
			}
		}
	}

	if (!first_row && headers.size() > 1) {
		bench_io_end();
		const char **MALLOCA(labels, headers.size() + 2);
		for (size_t i = 0; i < headers.size(); i++)
			labels[i] = headers[i].c_str();
		labels[headers.size()] = "No header line";
		pinfol("Which column contains your sequences?");
		header = pchoice(labels, headers.size() + 1,
				 "Enter column number");
		if (!(headers.size() - header)) {
			pinfol("Which column contains a sequence?");
			header = pchoice(labels, headers.size(),
					 "Enter column number");
		} else {
			first_row = 1;
		}
		free(labels);
		bench_io_start();
	}

	src.seqs.reserve(src.lines.size() - first_row);
	for (size_t row = first_row; row < src.lines.size(); row++) {
		auto line = src.lines[row];
		dsv_split(tokens, line, delimiter);
		if (tokens.size() != headers.size()) {
			perr("Column count mismatch at row %zu", row + 1);
			src.seqs.clear();
			return false;
		}

		auto token = tokens[header];
		if (token.empty()) {
			perr("Empty sequence found at row %zu", row + 1);
			src.seqs.clear();
			return false;
		}
		src.seqs.emplace_back(token);
	}

	if (src.seqs.empty()) {
		perr("No sequences found in DSV file");
		return false;
	}

	return true;
}

static format dsv_format(parse_dsv);
