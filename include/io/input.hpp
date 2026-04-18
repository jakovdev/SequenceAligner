#ifndef IO_INPUT_HPP
#define IO_INPUT_HPP

#include <string>
#include <string_view>
#include <vector>

std::string_view trim(std::string_view text) noexcept;

struct source {
	using parser_fn = bool (*)(source &) noexcept;
	inline static constinit std::vector<parser_fn> parsers{};
	std::vector<std::string_view> lines{};
	std::string extension{};
	std::vector<std::string> seqs{};

	[[gnu::nonnull]]
	explicit source(const char *path) noexcept;

	[[gnu::nonnull]]
	bool load(struct sequences *) noexcept;
};

struct format {
	format(source::parser_fn parser) noexcept
	{
		source::parsers.push_back(parser);
	}
};

#endif /* IO_INPUT_HPP */
