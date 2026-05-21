#ifndef IO_INPUT_HPP
#define IO_INPUT_HPP

#include <string>
#include <string_view>
#include <vector>

std::string_view trim(std::string_view text) noexcept;

struct source {
	enum class parse_result { SUCCESS, ERROR, UNSUPPORTED };
	using parser_fn = parse_result (*)(source &) noexcept;
	inline static constinit std::vector<parser_fn> parsers{};
	std::vector<std::string_view> lines{};
	std::string extension{};
	std::vector<std::string> seqs{};

	[[gnu::nonnull]]
	bool load(struct input *, const char *path) noexcept;
};

struct input_format {
	input_format(source::parser_fn parser) noexcept
	{
		source::parsers.push_back(parser);
	}
};

#endif /* IO_INPUT_HPP */
