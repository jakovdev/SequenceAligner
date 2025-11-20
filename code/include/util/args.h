#pragma once
#ifndef UTIL_ARGS_H
#define UTIL_ARGS_H

/**
  * @file args.h
  * @brief Decentralized and modular argument manager.
  * 
  * This header allows defining arguments in a distributed manner across the
  * codebase without a central list. Arguments are registered via constructors
  * that run before main(). Supported compilers include GCC, Clang, and MSVC.
  * 
  * Words inside backticks ("`") refer to grepp-able symbols or macros.
  * Words that start with a dot (".") refer to `struct argument` fields.
  * 
  * Basic usage:
  * 1. Define arguments in source files using the `ARGUMENT(name)` macro.
  * 2. Call `args_parse` in main().
  * 3. Call `args_validate` to run validators after everything is parsed.
  * 4. Call `args_actions` to run actions after everything is validated.
  */

#include <stdbool.h>

/**
  * @brief Defines and registers a new argument.
  * 
  * Use this macro in the global scope of a source file. It creates a
  * `struct argument` instance and registers it with the system.
  * 
  * Example:
  * @code
  * 	static bool verbose;
  * 	ARGUMENT(verbose) = {
  * 		.opt = 'v',
  * 		.lopt = "verbose",
  * 		.help = "Enable verbose output",
  * 		.set = &verbose,
  * 	}; @endcode
  * 
  * @param name Unique identifier for the argument.
  * 
  * 'name' gets prefixed so it does not clash with other symbols.
  * The prefixed symbol is global, so 'name' must be unique across the project.
  * 
  * `ARG(name)` can be used to reference it.
  */
#define ARGUMENT(name)                          \
	ARG_DECLARE(name);                      \
	_ARGS_CONSTRUCTOR(_arg_register_##name) \
	{                                       \
		_args_register(ARG(name));      \
	}                                       \
	ARG_DECLARE(name)

/**
  * @brief Forward declares an argument defined in the same file.
  * 
  * Useful if you need to reference the argument before its definition.
  * 
  * Example:
  * @code
  * 	ARG_DECLARE(foo);
  * 	ARGUMENT(bar) = {
  * 		...
  * 		ARG_DEPENDS(ARG(foo)),
  * 		...
  * 	};
  * 	ARGUMENT(foo) = { ... }; @endcode
  */
#define ARG_DECLARE(name) struct argument _arg_##name

/**
  * @brief Externally declares an argument defined in another file.
  * 
  * Useful for defining `relations` across files.
  * 
  * Example:
  * 
  * foo.c
  * @code
  * 	ARGUMENT(foo) = { ... }; @endcode
  * 
  * bar.c
  * @code
  * 	ARG_EXTERN(foo);
  * 	ARGUMENT(bar) = {
  * 		...
  * 		ARG_DEPENDS(ARG(foo)),
  * 		...
  * 	}; @endcode
  */
#define ARG_EXTERN(name) extern struct argument _arg_##name

/** @brief Address of an argument for `ARG_DEPENDS` and `ARG_CONFLICTS`. */
#define ARG(name) &_arg_##name

/**
  * @brief Parses command line arguments.
  * 
  * For every user specified argument:
  * @code
  * If argument with parameter is already set, error for repeated argument.
  * Runs `.parse_callback` if it exists.
  * Checks `relations` if `relation_phase` = `ARG_RELATION_PARSE`.
  * Sets `.set` to true (even implicitly allocated ones). @endcode
  * 
  * @return false on any error, true if parsing succeeded.
  */
bool args_parse(int argc, char *argv[]);

/**
  * @brief Must be called after `args_parse`.
  * 
  * For every `ARGUMENT`:
  * @code
  * Checks if required arguments are set if `.arg_req` = `ARG_REQUIRED`.
  * If a conflict argument is set, the (required) argument must not be set.
  * Checks `relations` if `relation_phase` != `ARG_RELATION_PARSE`.
  * Runs `.validate_callback` if it exists using `.validate_phase`. @endcode
  * 
  * @return false on any error, true if all arguments are valid.
  */
bool args_validate(void);

/**
  * @brief Must be called after `args_validate`. Optional if not using actions.
  * 
  * For every `ARGUMENT`:
  * @code
  * Runs `.action_callback` if it exists using `.action_phase`. @endcode
  */
void args_actions(void);

#ifdef ARGS_NO_DEFAULT_HELP
/**
 * @brief Prints the help message.
 * 
 * Exposed when `ARGS_NO_DEFAULT_HELP` is defined so you can call it from your
 * custom help argument.
 * 
 * If not defined, this function is static and called automatically by
 * the default ARGUMENT(help).
 * 
 * If defined, the ARGUMENT(help) will not be created automatically.
 */
void args_print_help(void);
#endif

#include <stddef.h>

/** @brief Return type for `.parse_callback` and `.validate_callback`. */
struct arg_callback {
	const char *error; /* NULL if valid. */
};

/** @brief Helper to return an invalid result with a message in a callback. */
#define ARG_INVALID(msg) ((struct arg_callback){ .error = msg })

/** @brief Helper to return a valid result in a callback. */
#define ARG_VALID() ((struct arg_callback){ .error = NULL })

/** @brief Values for `.param_req`. Follows getopt conventions. */
enum arg_parameter {
	ARG_PARAM_NONE,
	ARG_PARAM_REQUIRED,
	ARG_PARAM_OPTIONAL,
};

/** @brief Values for `.arg_req`. */
enum arg_requirement {
	ARG_OPTIONAL,
	ARG_REQUIRED,
	ARG_HIDDEN, /* Like OPTIONAL but hidden from help. */
	ARG_SOMETIME, /* Use `.validate_callback` or `relations`. */
};

/** @brief Values for `.validate_phase` and `.action_phase`. */
enum arg_callback_phase {
	ARG_CALLBACK_ALWAYS, /* Skips function if NULL */
	ARG_CALLBACK_IF_SET,
	ARG_CALLBACK_IF_UNSET,
};

/** @brief Values for `relation_phase` in `ARG_DEPENDS` and `ARG_CONFLICTS` */
enum arg_relation_phase {
	ARG_RELATION_PARSE, /* Skips `relations` if NULL */
	ARG_RELATION_VALIDATE_ALWAYS,
	ARG_RELATION_VALIDATE_SET,
	ARG_RELATION_VALIDATE_UNSET,
};

/* Internal state, do not modify directly. */
struct args_internal {
	struct argument *next_args;
	struct argument *next_help;
	struct argument *next_validate;
	struct argument *next_action;
	struct argument **deps;
	size_t deps_n;
	struct argument **cons;
	size_t cons_n;
	size_t help_len;
	enum arg_relation_phase deps_phase;
	enum arg_relation_phase cons_phase;
	bool valid;
};

#ifndef ARGS_IMPLICIT_SETS
/** @brief Maximum number of implicitly allocated .set booleans. */
#define ARGS_IMPLICIT_SETS (64)
#endif

/**
  * @brief Argument `relations`, a list of dependencies.
  * 
  * When setting this argument, dependencies must already be set.
  * 
  * Must be called inside an `ARGUMENT` definition.
  * 
  * Example:
  * @code
  * 	ARGUMENT(foo) = { ... };
  * 	ARGUMENT(bar) = { ... };
  * 	ARGUMENT(baz) = {
  * 		...
  * 		ARG_DEPENDS(ARG_RELATION_PARSE, ARG(foo), ARG(bar)),
  * 		...
  * 	}; @endcode
  * 
  * @param relation_phase When to check dependencies, see `arg_relation_phase`.
  */
#define ARG_DEPENDS(relation_phase, ...)                           \
	._.deps_phase = relation_phase,                            \
	._.deps = (struct argument *[]){ __VA_ARGS__, NULL },      \
	._.deps_n = sizeof((struct argument *[]){ __VA_ARGS__ }) / \
		    sizeof(struct argument *)

/**
  * @brief Argument `relations`, a list of conflicts.
  * 
  * If a conflict is set, this one must not be set, even if `ARG_REQUIRED`.
  * 
  * Must be called inside an `ARGUMENT` definition.
  * 
  * Example:
  * @code
  * 	ARGUMENT(foo) = { ... };
  * 	ARGUMENT(bar) = { ... };
  * 	ARGUMENT(baz) = {
  * 		...
  * 		ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(foo), ARG(bar)),
  * 		...
  * 	}; @endcode
  * 
  * @param relation_phase When to check conflicts, see `arg_relation_phase`.
  */
#define ARG_CONFLICTS(relation_phase, ...)                         \
	._.cons_phase = relation_phase,                            \
	._.cons = (struct argument *[]){ __VA_ARGS__, NULL },      \
	._.cons_n = sizeof((struct argument *[]){ __VA_ARGS__ }) / \
		    sizeof(struct argument *)

/**
  * @brief Configuration structure for an argument.
  * 
  * Initialize this struct using the `ARGUMENT(name)` macro.
  * 
  * Most fields are optional and should default to 0 or NULL using C's
  * partial struct initialization when not explicitly provided.
  */
struct argument {
	/**
	  * @brief Indicates if the argument was provided by the user.
	  * 
	  * If NULL, it is implicitly allocated internally if needed.
	  * 
	  * Explicitly point this to something if you need it elsewhere.
	  * Best used for "flag" arguments with no parameter.
	  * 
	  * You can use both `.set` and `.dest` for more complex scenarios,
	  * though `relations` should cover most of them.
	  */
	bool *set;

	/** @brief Destination variable for `.parse_callback`. Can be NULL */
	void *dest;

	/**
	  * @brief Function that is called during `args_parse`.
	  * 
	  * Use `ARG_PARSER` macros for convenience.
	  * 
	  * Doesn't need to be a "parsing" function, so `.dest` can be NULL.
	  * 
	  * Also useful for exiting arguments like '--help' or '--version'.
	  * 
	  * Example:
	  * @code
  * 	static double mydouble;
  * 	// Creates parse_mydoubles() for .parse_callback
  * 	ARG_PARSE_D(mydoubles, double, , val < 5.0, "Must be >= 5.0");
  * 	ARGUMENT(myarg) = {
  * 		...
  * 		.dest = &mydouble,
  * 		.parse_callback = parse_mydoubles,
  * 		...
  * 	}; @endcode
	  * 
	  * You can also write custom parsers:
	  * @code
  * 	enum Color { COLOR_INVALID = -1, RED, GREEN, BLUE };
  * 	static enum Color color = COLOR_INVALID;
  * 	static struct arg_callback parse_color(const char *str, void *dest) {
  * 		// Using 'color' directly is also possible in this example
  * 		enum Color col = COLOR_INVALID;
  * 		if (strcmp(str, "red") == 0)
  * 			col = RED;
  * 		else if (strcmp(str, "green") == 0)
  * 			col = GREEN;
  * 		else if (strcmp(str, "blue") == 0)
  * 			col = BLUE;
  * 		else
  * 			return ARG_INVALID("Invalid color");
  * 		*(enum Color *)dest = col; // dest points to color
  * 		return ARG_VALID();
  * 	}
  * 	ARGUMENT(color) = {
  * 		...
  * 		.dest = &color,
  * 		.parse_callback = parse_color,
  * 		...
  * 	}; @endcode
	  * 
	  * @param str The string value provided by the user.
	  * @param dest Pointer to `.dest` which could be NULL, up to you.
	  * @return `ARG_INVALID(msg)` on error, `ARG_VALID()` on success.
	  */
	struct arg_callback (*parse_callback)(const char *str, void *dest);

	/**
	  * @brief Function that is called during the `.validate_phase`.
	  * 
	  * Use for complex checks.
	  * 
	  * Example:
	  * @code
  * 	enum Color { RED, GREEN, BLUE };
  * 	static enum Color color; // preferably set with a custom parser
  * 	static bool other_option; // could be anything, just an example
  * 	static struct arg_callback validate_color(void) {
  * 		if (color == RED && other_option)
  * 			return ARG_INVALID("Red color cannot be used.");
  * 		return ARG_VALID();
  * 	}
  * 	ARGUMENT(color) = {
  * 		...
  * 		.dest = &color,
  * 		.validate_callback = validate_color,
  * 		...
  * 	}; @endcode
	  * 
	  * @return `ARG_INVALID(msg)` on error, `ARG_VALID()` on success.
	  */
	struct arg_callback (*validate_callback)(void);

	/**
	  * @brief Function that is called during the `.action_phase`.
	  * 
	  * Use for side effects, like configuration printing.
	  * 
	  * Example:
	  * @code
  * 	static bool verbose;
  * 	static void print_config(void) {
  * 		printf("Verbose mode is on");
  * 	}
  * 	ARGUMENT(verbose) = {
  * 		...
  * 		.set = &verbose,
  * 		.action_callback = print_config,
  * 		.action_phase = ARG_CALLBACK_IF_SET,
  * 		...
  * 	}; @endcode
	  */
	void (*action_callback)(void);

	/** @brief Is the argument required? */
	enum arg_requirement arg_req;
	/** @brief Does it take a parameter? */
	enum arg_parameter param_req;

	/** @brief When to run `.validate_callback`. */
	enum arg_callback_phase validate_phase;
	/** @brief When to run `.action_callback`. */
	enum arg_callback_phase action_phase;

	/**
	  * @brief Help description when ARGUMENT(help) is specified.
	  * 
	  * Multiline strings are supported and will be indented automatically.
	  * 
	  * Example: "First line.\nSecond line."
	  * @code
  * NOTE: Might be more obvious with a monospaced font.
  * 	-o, --option N  First line.
  * 	                Second line. @endcode
	  */
	const char *help;

	/**
	  * @brief Parameter description when ARGUMENT(help) is specified.
	  * 
	  * Printed right after opt, lopt in help message.
	  * 
	  * Example: "N" for an argument that takes a number.
	  * 
	  * Required if `.param_req` != `ARG_PARAM_NONE`.
	  */
	const char *param;

	/** @brief Long option (e.g., "output" for --output). */
	const char *lopt;
	/** @brief Short option (e.g., 'o' for -o). */
	char opt;

	/**
	  * @brief Sort weight for help display.
	  * 
	  * Higher weight = higher priority (printed first).
	  */
	unsigned int help_weight;

	/**
	  * @brief Sort weight for validation.
	  * 
	  * Higher weight = higher priority (validated first).
	  */
	unsigned int validate_weight;

	/**
	  * @brief Sort weight for action.
	  * 
	  * Higher weight = higher priority (executed first).
	  */
	unsigned int action_weight;

	struct args_internal _; /* Internal use only. */
};

/** @brief Global raw arguments (argc/argv) access. */
extern struct args_raw {
	int c;
	char **v;
} argr;

/** @brief For use in `ARG_PARSER` `ARG_BASE` parameter. */
#define ARG_BASE(N) , N

#include <errno.h>

/**
  * @brief Generates a parser function.
  * 
  * See convenience macros for common functions like strtod or strtol below.
  * 
  * See example usage in `.parse_callback` documentation.
  * 
  * @param name Name of the parser (generates 'parse_name').
  * @param strto Standard-compatible string conversion function.
  * @param ARG_BASE Base for conversion (e.g., 'ARG_BASE(10)'), or empty.
  * @param strto_t Return type of strto function.
  * @param dest_t Type of the destination variable.
  * @param CAST Cast if strto_t != dest_t (e.g., '(int)'), or empty if same.
  * @param cond Boolean expression using 'val'. If true, the value is invalid.
  * @param err Error message if conversion fails.
  */
#define ARG_PARSER(name, strto, ARG_BASE, strto_t, dest_t, CAST, cond, err)  \
	static struct arg_callback parse_##name(const char *str, void *dest) \
	{                                                                    \
		errno = 0;                                                   \
		char *endptr = NULL;                                         \
		strto_t val = strto(str, &endptr ARG_BASE);                  \
		if (endptr == str || *endptr != '\0' || errno == ERANGE ||   \
		    (cond)) {                                                \
			if (!err)                                            \
				return ARG_INVALID("Invalid value");         \
			else                                                 \
				return ARG_INVALID(err);                     \
		}                                                            \
		*(dest_t *)dest = CAST val;                                  \
		return ARG_VALID();                                          \
	}

/** @brief Convenience macro for 'strtol'. */
#define ARG_PARSE_L(name, base, dest_t, CAST, cond, err) \
	ARG_PARSER(name, strtol, ARG_BASE(base), long, dest_t, CAST, cond, err)

/** @brief Convenience macro for 'strtoll'. */
#define ARG_PARSE_LL(name, base, dest_t, CAST, cond, err)                  \
	ARG_PARSER(name, strtoll, ARG_BASE(base), long long, dest_t, CAST, \
		   cond, err)

/** @brief Convenience macro for 'strtoul'. */
#define ARG_PARSE_UL(name, base, dest_t, CAST, cond, err)                      \
	ARG_PARSER(name, strtoul, ARG_BASE(base), unsigned long, dest_t, CAST, \
		   cond, err)

/** @brief Convenience macro for 'strtoull'. */
#define ARG_PARSE_ULL(name, base, dest_t, CAST, cond, err)                     \
	ARG_PARSER(name, strtoull, ARG_BASE(base), unsigned long long, dest_t, \
		   CAST, cond, err)

/** @brief Convenience macro for 'strtof'. */
#define ARG_PARSE_F(name, dest_t, CAST, cond, err) \
	ARG_PARSER(name, strtof, , float, dest_t, CAST, cond, err)

/** @brief Convenience macro for 'strtod'. */
#define ARG_PARSE_D(name, dest_t, CAST, cond, err) \
	ARG_PARSER(name, strtod, , double, dest_t, CAST, cond, err)

void _args_register(struct argument *a);

#ifdef _MSC_VER
#define _ARGS_CRT_INIT ".CRT$XCU"
#define _ARGS_SECTION_READ(sect) section(sect, read)
#pragma _ARGS_SECTION_READ(_ARGS_CRT_INIT)

#define _ARGS_PRAGMA(x) __pragma(x)
#define _ARGS_DECLSPEC(x) __declspec(x)
#define _ARGS_ALLOCATE(x) _ARGS_DECLSPEC(allocate(x))
#define _ARGS_LINKER(x) _ARGS_PRAGMA(comment(linker, "/include:" x))

/* Use this to run functions before main() */
#define _ARGS_CONSTRUCTOR(name)                   \
	static void name(void);                   \
	_ARGS_LINKER(_##name)                     \
	_ARGS_ALLOCATE(_ARGS_CRT_INIT)            \
	static void (*_##name##_fp)(void) = name; \
	static void name(void)
#else /* GCC/Clang */
/* Use this to run functions before main() */
#define _ARGS_CONSTRUCTOR(name)                              \
	static void name(void) __attribute__((constructor)); \
	static void name(void)
#endif

#endif /* UTIL_ARGS_H */
