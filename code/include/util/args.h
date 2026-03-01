#pragma once
#ifndef UTIL_ARGS_H
#define UTIL_ARGS_H

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>

/**
  * @file args.h
  * @brief Decentralized and modular argument manager.
  * 
  * This header allows defining arguments in a distributed manner across the
  * codebase without a central list. Arguments are registered via constructors
  * that run before main(). Supported compilers include GCC, Clang, and MSVC.
  * Words inside backticks ("`") refer to grepp-able symbols or macros.
  * Words that start with a dot (".") refer to `struct argument` fields.
  * 
  * Basic usage:
  * 1. Define arguments in source files using the `ARGUMENT(name)` macro.
  * 2. #define ARGS_IMPLEMENTATION in main.c and include this header.
  * 3. Call `args_parse`, `args_validate`, and optionally `args_actions`.
  * 
  * C's partial struct initialization means you only need to specify the fields
  * you care about as defaults will handle the rest.
  */

/*-------------------------*/
/* CREATING A NEW ARGUMENT */
/*-------------------------*/

#ifndef __cplusplus
/**
  * @brief Creates and registers a new argument.
  * 
  * Use this macro in the global scope of a source file. It creates a
  * `struct argument` instance and registers it.
  * 
  * Example:
  * @code{.c}
  * static bool verbose;
  * ARGUMENT(verbose) = {
  * 	.opt = 'v',
  * 	.lopt = "verbose",
  * 	.help = "Enable verbose output",
  * 	.set = &verbose,
  * };
  * @endcode
  * 
  * @param name Unique identifier for the argument
  */
#define ARGUMENT(name)                          \
	ARG_DECLARE(name);                      \
	_ARGS_CONSTRUCTOR(_arg_register_##name) \
	{                                       \
		_args_register(ARG(name));      \
	}                                       \
	ARG_DECLARE(name)
#else
#define ARGUMENT(name)                          \
	ARG_EXTERN(name);                       \
	_ARGS_CONSTRUCTOR(_arg_register_##name) \
	{                                       \
		_args_register(ARG(name));      \
	}                                       \
	ARG_DECLARE(name)
#endif

/* ARGUMENT REQUIREMENT LEVELS / VISIBILITY */

/** @brief Values for `.arg_req`. */
enum arg_requirement {
	ARG_OPTIONAL, /* Default: user doesn't need to specify */
	ARG_REQUIRED, /* User must specify */
	ARG_HIDDEN, /* Optional but hidden from help */
	ARG_SOMETIME, /* Custom, use `.validate_callback` or `relations` */
};

/*-----------------------*/
/* NON-BOOLEAN ARGUMENTS */
/*-----------------------*/

/** @brief Values for `.param_req`. Follows getopt conventions. */
enum arg_parameter {
	ARG_PARAM_NONE,
	ARG_PARAM_REQUIRED, /* In `.parse_callback` `str` will not be NULL */
	ARG_PARAM_OPTIONAL, /* In `.parse_callback` `str` can be NULL */
};

/* Predefined parsers: ARG_PARSE_L, ARG_PARSE_LL, ARG_PARSE_UL, ARG_PARSE_ULL,
 * ARG_PARSE_F, ARG_PARSE_D. Create custom parsers with ARG_PARSER macro.
 */

/**
  * @brief Generates a parser function.
  * 
  * Convenience macros `ARG_PARSE_*` wrap this for common types.
  * 
  * @param name Name of the parser (generates 'parse_name').
  * @param strto String conversion function (strtol, strtod, etc.).
  * @param ARG_BASE Base for conversion (e.g., `ARG_BASE(`10`)`) or empty.
  * @param strto_t Return type of strto function.
  * @param dest_t Destination variable type.
  * @param CAST Cast if types differ, or empty.
  * @param cond Boolean condition using 'val' for rejection.
  * @param err Error message on failure.
  */
#define ARG_PARSER(name, strto, ARG_BASE, strto_t, dest_t, CAST, cond, err)    \
	static struct arg_callback parse_##name(const char *str, void *dest)   \
	{                                                                      \
		errno = 0;                                                     \
		char *end = NULL;                                              \
		strto_t val = strto(str, &end ARG_BASE);                       \
		if (end == str || *end != '\0' || errno == ERANGE || (cond))   \
			return ARG_INVALID((err) && *(err) ? (err) : ARG_ERR); \
		*(dest_t *)dest = CAST val;                                    \
		return ARG_VALID();                                            \
	}

/** @brief For use in `ARG_PARSER` `ARG_BASE` parameter. */
#define ARG_BASE(N) , N

#ifndef ARG_ERR
/** @brief Default error message for parsers. `OVERRIDABLE` */
#define ARG_ERR "Invalid value"
#endif

/** @brief Convenience macro for 'strtol'. Parameters match `ARG_PARSER`. */
#define ARG_PARSE_L(name, base, dest_t, CAST, cond, err) \
	ARG_PARSER(name, strtol, ARG_BASE(base), long, dest_t, CAST, cond, err)

/** @brief Convenience macro for 'strtoll'. Parameters match `ARG_PARSER`. */
#define ARG_PARSE_LL(name, base, dest_t, CAST, cond, err)                  \
	ARG_PARSER(name, strtoll, ARG_BASE(base), long long, dest_t, CAST, \
		   cond, err)

/** @brief Convenience macro for 'strtoul'. Parameters match `ARG_PARSER`. */
#define ARG_PARSE_UL(name, base, dest_t, CAST, cond, err)                      \
	ARG_PARSER(name, strtoul, ARG_BASE(base), unsigned long, dest_t, CAST, \
		   cond, err)

/** @brief Convenience macro for 'strtoull'. Parameters match `ARG_PARSER`. */
#define ARG_PARSE_ULL(name, base, dest_t, CAST, cond, err)                     \
	ARG_PARSER(name, strtoull, ARG_BASE(base), unsigned long long, dest_t, \
		   CAST, cond, err)

/** @brief Convenience macro for 'strtof'. Parameters match `ARG_PARSER`. */
#define ARG_PARSE_F(name, dest_t, CAST, cond, err) \
	ARG_PARSER(name, strtof, , float, dest_t, CAST, cond, err)

/** @brief Convenience macro for 'strtod'. Parameters match `ARG_PARSER`. */
#define ARG_PARSE_D(name, dest_t, CAST, cond, err) \
	ARG_PARSER(name, strtod, , double, dest_t, CAST, cond, err)

/* Custom parsers must return `ARG_INVALID(msg)` or `ARG_VALID()`, with
 * signature (const char *str, void *dest).
 */

/** @brief Return type for `.parse_callback` and `.validate_callback`. */
struct arg_callback {
	const char *error; /* NULL if valid. */
};

/** @brief Helper to return an invalid result with a message in a callback. */
#define ARG_INVALID(msg) ((struct arg_callback){ .error = msg })

/** @brief Helper to return a valid result in a callback. */
#define ARG_VALID() ((struct arg_callback){ .error = NULL })

/** @brief Raw arguments (argc/argv) struct. Useful for e.g. binary path */
struct args_raw {
	int c;
	char **v;
};

/** @brief Define if you want argr (argc/argv) in your source file. */
#ifdef ARGS_GLOBAL_ARGR
extern struct args_raw argr;
#endif

/* CALLING PARSERS FOR ALL ARGUMENTS */

/**
  * @brief Parses command line arguments.
  * 
  * For every user specified argument:
  * - If argument with parameter is already set, error for repeated argument.
  * - Runs `.parse_callback` if it exists.
  * - Checks `relations` if `relation_phase` = `ARG_RELATION_PARSE`.
  * - Sets `.set` to true (even implicitly allocated ones).
  * 
  * @return false on error, true on success.
  */
bool args_parse(int argc, char *argv[]);

/*----------------------*/
/* CALLBACKS AND PHASES */
/*----------------------*/

/* Execution flows through parse, validate, and action stages.
 * Each stage has phases, callbacks and ordering.
 * Internal logic (how you defined an argument) is checked within them.
 * External logic related to your project should be handled in your callbacks.
 */

/** @brief Values for `.validate_phase` and `.action_phase`. */
enum arg_callback_phase {
	ARG_CALLBACK_ALWAYS, /* Always run (skips if callback is NULL) */
	ARG_CALLBACK_IF_SET, /* Only if argument was provided by user */
	ARG_CALLBACK_IF_UNSET, /* Only if argument was not provided by user */
};

/**
  * @brief Must be called after `args_parse`.
  * 
  * For every `ARGUMENT`:
  * - Checks if required arguments are set if `.arg_req` = `ARG_REQUIRED`.
  * - If a conflict argument is set, the (required) argument must not be set.
  * - Checks `relations` if `relation_phase` != `ARG_RELATION_PARSE`.
  * - Runs `.validate_callback` if it exists using `.validate_phase`.
  * 
  * @return false on error, true on success.
  */
bool args_validate(void);

/**
  * @brief Must be called after `args_validate`. Optional if not using actions.
  * 
  * For every `ARGUMENT`:
  * - Runs `.action_callback` if it exists using `.action_phase`.
  */
void args_actions(void);

/*------------------------*/
/* ARGUMENT RELATIONSHIPS */
/*------------------------*/

/* Relations: dependencies (require other args), conflicts (exclude others),
 * and subsets (superset args trigger multiple subset args from one).
 * Use `ARG(name)` to reference arguments in relations.
 */

/** @brief Address of an argument for `ARG_DEPENDS` and `ARG_CONFLICTS`. */
#define ARG(name) &_arg_##name

/**
  * @brief Forward declares an argument for same-file forward references.
  * 
  * Example:
  * @code{.c}
  * ARG_DECLARE(foo); // Can now be referenced after this point.
  * ARGUMENT(bar) = {
  * 	...
  * 	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(foo)),
  * 	...
  * };
  * ARGUMENT(foo) = { ... };
  * @endcode
  */
#define ARG_DECLARE(name) struct argument _arg_##name

/**
  * @brief Externally declares an argument from another file.
  * 
  * Example:
  * 
  * foo.c
  * @code{.c}
  * ARGUMENT(foo) = { ... };
  * @endcode
  * 
  * bar.c
  * @code{.c}
  * ARG_EXTERN(foo); // Can now be referenced after this point.
  * ARGUMENT(bar) = {
  * 	...
  * 	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(foo)),
  * 	...
  * };
  * @endcode
  */
#define ARG_EXTERN(name) extern struct argument _arg_##name

/**
  * @brief Specify argument dependencies, a `relation`.
  * 
  * When setting this argument, all dependencies must already be set.
  * Use inside an `ARGUMENT` definition.
  * 
  * Example:
  * @code{.c}
  * ARGUMENT(foo) = { ... };
  * ARGUMENT(bar) = { ... };
  * ARGUMENT(baz) = {
  * 	...
  * 	ARG_DEPENDS(ARG_RELATION_PARSE, ARG(foo), ARG(bar)),
  * 	...
  * };
  * @endcode
  * 
  * @param relation_phase When to check dependencies, see `arg_relation_phase`.
  */
#define ARG_DEPENDS(relation_phase, ...)                           \
	._.deps_phase = relation_phase,                            \
	._.deps = (struct argument *[]){ __VA_ARGS__, NULL },      \
	._.deps_n = sizeof((struct argument *[]){ __VA_ARGS__ }) / \
		    sizeof(struct argument *)

/**
  * @brief Specify argument conflicts, a `relation`.
  * 
  * If a conflict argument is set, this one must not be set, overriding
  * ARG_REQUIRED. Use inside an `ARGUMENT` definition.
  * 
  * Example:
  * @code{.c}
  * ARGUMENT(foo) = { ... };
  * ARGUMENT(bar) = { ... };
  * ARGUMENT(baz) = {
  * 	...
  * 	ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(foo), ARG(bar)),
  * 	...
  * };
  * @endcode
  * 
  * @param relation_phase When to check conflicts, see `arg_relation_phase`.
  */
#define ARG_CONFLICTS(relation_phase, ...)                         \
	._.cons_phase = relation_phase,                            \
	._.cons = (struct argument *[]){ __VA_ARGS__, NULL },      \
	._.cons_n = sizeof((struct argument *[]){ __VA_ARGS__ }) / \
		    sizeof(struct argument *)

/**
  * @brief Specify subset arguments, a `relation`.
  * 
  * When this argument is set, all subsets are also processed. Parent string
  * is passed to subset parsers unless customized with `ARG_SUBSTRINGS`.
  * Use inside an `ARGUMENT` definition.
  * 
  * Example:
  * @code{.c}
  * ARGUMENT(foo) = { ... };
  * ARGUMENT(bar) = { ... };
  * ARGUMENT(baz) = {
  * 	...
  * 	ARG_SUBSETS(ARG(foo), ARG(bar)),
  * 	ARG_SUBSTRINGS("out.txt", ARG_SUBPASS),
  * 	...
  * };
  * @endcode
  */
#define ARG_SUBSETS(...)                                           \
	._.subs = (struct argument *[]){ __VA_ARGS__, NULL },      \
	._.subs_n = sizeof((struct argument *[]){ __VA_ARGS__ }) / \
		    sizeof(struct argument *)

/**
  * @brief Custom strings for index-aligned subset arguments.
  * 
  * Use `ARG_SUBPASS` to pass parent string, omit entirely if not needed.
  */
#define ARG_SUBSTRINGS(...) \
	._.subs_strs = ((const char *[]){ __VA_ARGS__, NULL })

/** @brief Signal to pass the parent string */
#define ARG_SUBPASS ((const char *)-1)

/** @brief Values for `relation_phase` in `ARG_DEPENDS` and `ARG_CONFLICTS` */
enum arg_relation_phase {
	ARG_RELATION_PARSE, /* Skips `relations` if NULL */
	ARG_RELATION_VALIDATE_ALWAYS,
	ARG_RELATION_VALIDATE_SET,
	ARG_RELATION_VALIDATE_UNSET,
};

/*---------------------------------------------------*/
/* DETERMINISTIC AND CUSTOM ARGUMENT EXECUTION ORDER */
/*---------------------------------------------------*/

/* Arguments in the same file process in declaration order. Use *`_order`
 * fields to control execution order across files.
 */

/** @brief Use in `validate_order`, `action_order`, or `help_order`. */
#define ARG_ORDER_FIRST ((struct argument *)-1)

/**
  * @brief Use in `validate_order`, `action_order`, or `help_order`.
  *
  * Use `ARG_DECLARE` or `ARG_EXTERN` to forward-declare the referenced arg.
  */
#define ARG_ORDER_AFTER(arg_name) (ARG(arg_name))

/* Internal state, do not modify directly. Skip to below. */
struct args_internal {
	struct argument *next_args;
	struct argument *next_help;
	struct argument *next_validate;
	struct argument *next_action;
	struct argument **deps;
	size_t deps_n;
	struct argument **cons;
	size_t cons_n;
	struct argument **subs;
	const char **subs_strs;
	size_t subs_n;
	size_t help_len;
	enum arg_relation_phase deps_phase;
	enum arg_relation_phase cons_phase;
	bool valid;
};

/**
  * @brief Configuration structure for an argument.
  * 
  * Initialize this struct using the `ARGUMENT(name)` macro.
  * Most fields are optional and should default to 0 or NULL using C's
  * partial struct initialization when not explicitly provided.
  */
struct argument {
	/**
	  * @brief Indicates if the argument was provided by the user.
	  * 
	  * If NULL, allocated implicitly if needed. Explicitly set it to track
	  * the argument's presence elsewhere in your code.
	  */
	bool *set;

	/** @brief Destination variable for `.parse_callback`. Can be NULL */
	void *dest;

	/**
	  * @brief Function that is called during `args_parse`.
	  * 
	  * Use `ARG_PARSER` macros for convenience.
	  * Doesn't need to be a "parsing" function, so `.dest` can be NULL.
	  * Also useful for exiting arguments like '--help' or '--version'.
	  * 
	  * Example:
	  * @code{.c}
	  * static double mydouble;
	  * // Creates parse_mydoubles() for .parse_callback
	  * ARG_PARSE_D(mydoubles, double, , val < 5.0, "Must be >= 5.0");
	  * ARGUMENT(myarg) = {
	  * 	...
	  * 	.dest = &mydouble,
	  * 	.parse_callback = parse_mydoubles,
	  * 	...
	  * };
	  * @endcode
	  * 
	  * You can also write custom parsers:
	  * @code{.c}
	  * enum Color { COLOR_INVALID = -1, RED, GREEN, BLUE };
	  * static enum Color color = COLOR_INVALID;
	  * static struct arg_callback parse_color(const char *str, void *dest) {
	  * 	// Using 'color' directly is also possible in this example
	  * 	enum Color col = COLOR_INVALID;
	  * 	if (strcmp(str, "red") == 0)
	  * 		col = RED;
	  * 	else if (strcmp(str, "green") == 0)
	  * 		col = GREEN;
	  * 	else if (strcmp(str, "blue") == 0)
	  * 		col = BLUE;
	  * 	else
	  * 		return ARG_INVALID("Invalid color");
	  * 	*(enum Color *)dest = col; // dest points to color
	  * 	return ARG_VALID();
	  * }
	  * ARGUMENT(color) = {
	  * 	...
	  * 	.dest = &color,
	  * 	.parse_callback = parse_color,
	  * 	...
	  * };
	  * @endcode
	  * 
	  * @param str The string value provided by the user.
	  * @param dest Pointer to `.dest` which could be NULL, up to you.
	  * @return `ARG_INVALID(msg)` on error, `ARG_VALID()` on success.
	  */
	struct arg_callback (*parse_callback)(const char *str, void *dest);

	/**
	  * @brief Function that is called during the `.validate_phase`.
	  * 
	  * Use for complex checks after everything is parsed.
	  * 
	  * Example:
	  * @code{.c}
	  * enum Color { RED, GREEN, BLUE };
	  * static enum Color color;  // preferably set with a custom parser
	  * static bool other_option; // could be anything, just an example
	  * static struct arg_callback validate_color(void) {
	  * 	if (color == RED && other_option)
	  * 		return ARG_INVALID("Red color cannot be used.");
	  * 	return ARG_VALID();
	  * }
	  * ARGUMENT(color) = {
	  * 	...
	  * 	.dest = &color,
	  * 	.validate_callback = validate_color,
	  * 	...
	  * };
	  * @endcode
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
	  * @code{.c}
	  * static bool verbose;
	  * static void print_config(void) {
	  * 	printf("Verbose mode is on");
	  * }
	  * ARGUMENT(verbose) = {
	  * 	...
	  * 	.set = &verbose,
	  * 	.action_callback = print_config,
	  * 	.action_phase = ARG_CALLBACK_IF_SET,
	  * 	...
	  * };
	  * @endcode
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
	  * @brief Ordering for validation.
	  * 
	  * Use `ARG_ORDER_FIRST` or `ARG_ORDER_AFTER(arg_name)`, NULL = last.
	  */
	struct argument *validate_order;

	/**
	  * @brief Ordering for action execution.
	  * 
	  * Use `ARG_ORDER_FIRST` or `ARG_ORDER_AFTER(arg_name)`, NULL = last.
	  */
	struct argument *action_order;

	/**
	  * @brief Ordering for help display.
	  * 
	  * Use `ARG_ORDER_FIRST` or `ARG_ORDER_AFTER(arg_name)`, NULL = last.
	  */
	struct argument *help_order;

	/**
	  * @brief Help description for this argument.
	  * 
	  * Multiline strings supported (e.g., "Line 1.\nLine 2.").
	  * See: `ARGS_STR_PREPAD`, `ARGS_PARAM_OFFSET`, `ARGS_HELP_OFFSET`.
	  */
	const char *help;

	/**
	  * @brief Parameter name for help display (e.g., "N" for a number).
	  * 
	  * Required if `.param_req` != `ARG_PARAM_NONE`.
	  */
	const char *param;

	/** @brief Long option (e.g., "output" for --output). */
	const char *lopt;
	/** @brief Short option (e.g., 'o' for -o). */
	char opt;

	struct args_internal _; /* Internal use only. */
};

#ifdef ARGS_NO_DEFAULT_HELP /* `OVERRIDABLE` */
/**
 * @brief Prints the help message.
 * 
 * Only exposed when `ARGS_NO_DEFAULT_HELP` is defined, allowing custom
 * help argument implementation.
 * Otherwise, called automatically by default help argument.
 */
void args_print_help(void);
#endif

/** @brief Internal function, use `ARGUMENT` macro instead. */
void _args_register(struct argument *);

/* https://stackoverflow.com/questions/1113409/attribute-constructor-equivalent-in-vc */
#ifdef __cplusplus
#define _ARGS_CONSTRUCTOR(f) \
	static void f(void); \
	struct f##_t_ {      \
		f##_t_(void) \
		{            \
			f(); \
		}            \
	};                   \
	static f##_t_ f##_;  \
	static void f(void)
#elif defined(_MSC_VER) && !defined(__clang__)
#pragma section(".CRT$XCU", read)
#define _ARGS_CONSTRUCTOR2_(f, p)                                \
	static void f(void);                                     \
	__declspec(allocate(".CRT$XCU")) void (*f##_)(void) = f; \
	__pragma(comment(linker, "/include:" p #f "_")) static void f(void)
#ifdef _WIN64
#define _ARGS_CONSTRUCTOR(f) _ARGS_CONSTRUCTOR2_(f, "")
#else /* _WIN32 */
#define _ARGS_CONSTRUCTOR(f) _ARGS_CONSTRUCTOR2_(f, "_")
#endif
#else /* GCC, Clang */
#define _ARGS_CONSTRUCTOR(f)                              \
	static void f(void) __attribute__((constructor)); \
	static void f(void)
#endif

#ifdef ARGS_IMPLEMENTATION

#ifndef args_pe
/** @brief Error print. `OVERRIDABLE`. */
#define args_pe(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef args_pd
/** @brief Developer-only debug print. `OVERRIDABLE`. */
#define args_pd(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef args_pi
/** @brief Internal error print, user facing dev print. `OVERRIDABLE`. */
#define args_pi(arg) args_pe("Internal error for %s", arg_str(arg))
#endif

#ifndef args_abort
/** @brief Abort function. `OVERRIDABLE`. */
#define args_abort() abort()
#endif

#ifndef ARGS_STR_PREPAD
/** @brief Pre-padding for help text. `OVERRIDABLE`. */
#define ARGS_STR_PREPAD (2)
#elif ARGS_STR_PREPAD < 0
#error ARGS_STR_PREPAD cannot be negative
#endif

#ifndef ARGS_PARAM_OFFSET
/** @brief Offset between argument and parameter in help text. `OVERRIDABLE`. */
#define ARGS_PARAM_OFFSET (1)
#elif ARGS_PARAM_OFFSET < 1
#error ARGS_PARAM_OFFSET must be at least 1 for proper formatting
#endif

#ifndef ARGS_HELP_OFFSET
/** @brief Offset from longest argument for help text. `OVERRIDABLE`. */
#define ARGS_HELP_OFFSET (4)
#elif ARGS_HELP_OFFSET < 1
#error ARGS_HELP_OFFSET must be at least 1 for proper formatting
#endif

#ifndef ARGS_IMPLICIT_SETS
/** @brief Maximum implicit allocations of .set booleans. `OVERRIDABLE`. */
#define ARGS_IMPLICIT_SETS (64)
#elif ARGS_IMPLICIT_SETS < 1
#error ARGS_IMPLICIT_SETS must be at least 1 for defined behavior
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct args_raw argr = { 0 };

/* Lists */
static struct argument *args;
static struct argument *help;
static struct argument *validate;
static struct argument *action;

#define for_each_arg(a, list) \
	for (struct argument *a = (list); a; a = (a)->_.next_##list)

#define for_each_rel(a, rel, var)                                  \
	for (size_t var##i = 0; var##i < (a)->_.rel##_n; var##i++) \
		for (struct argument *var = (a)->_.rel[var##i]; var; var = NULL)

static size_t args_num;
static size_t longest;

#ifndef ARG_STR_MAX_CALLS
#define ARG_STR_MAX_CALLS (2)
#endif

#ifndef ARG_STR_BUF_SIZE
#define ARG_STR_BUF_SIZE (BUFSIZ)
#endif

static const char *arg_str(const struct argument *a)
{
	if (!a)
		return "<null-arg>";

	static char buf[ARG_STR_MAX_CALLS][ARG_STR_BUF_SIZE];
	static size_t i = 0;

	if (a->opt && a->lopt)
		snprintf(buf[i], sizeof(buf[i]), "-%c, --%s", a->opt, a->lopt);
	else if (a->opt) /* TODO: Handle case where every arg has no lopt */
		snprintf(buf[i], sizeof(buf[i]), "-%c     ", a->opt);
	else if (a->lopt)
		snprintf(buf[i], sizeof(buf[i]), "    --%s", a->lopt);
	else
		return "<invalid-arg>";

	i = 1 - i;
	return buf[1 - i];
}

static void arg_set_new(struct argument *a)
{
	static bool sets[ARGS_IMPLICIT_SETS] = { 0 };
	static size_t sets_n = 0;

	if (sets_n >= ARGS_IMPLICIT_SETS) {
		args_pd("ARGS_IMPLICIT_SETS exceeded, try increasing it");
		args_pi(a);
		args_abort();
	}

	a->set = &sets[sets_n++];
}

void _args_register(struct argument *a)
{
	if (!a) {
		args_pd("Cannot register %s", arg_str(a));
		args_pi(a);
		args_abort();
	}

	if (!a->opt && !a->lopt) {
		args_pd("%s must have an option", arg_str(a));
		args_pi(a);
		args_abort();
	}

	if (a->_.valid) {
		args_pd("%s has internals pre-set", arg_str(a));
		args_pi(a);
		args_abort();
	}

	if (a->param_req != ARG_PARAM_NONE && !a->param) {
		args_pd("%s requires parameter but .param=NULL", arg_str(a));
		args_pi(a);
		args_abort();
	}

	if (a->param_req != ARG_PARAM_NONE && !a->parse_callback) {
		args_pd("%s has .param but .parse_callback=NULL", arg_str(a));
		args_pi(a);
		args_abort();
	}

	if (a->validate_phase != ARG_CALLBACK_ALWAYS && !a->validate_callback) {
		args_pd("%s has .validate_phase but .validate_callback=NULL",
			arg_str(a));
		args_pi(a);
		args_abort();
	}

	if (a->action_phase != ARG_CALLBACK_ALWAYS && !a->action_callback) {
		args_pd("%s has .action_phase but .action_callback=NULL",
			arg_str(a));
		args_pi(a);
		args_abort();
	}

	if (a->arg_req == ARG_SOMETIME && !a->_.deps && !a->_.cons &&
	    !a->validate_callback) {
		args_pd("%s has no dependencies, conflicts, or validator",
			arg_str(a));
		args_pi(a);
		args_abort();
	}

	if (!a->set) {
		bool needs_set = false;
		if (a->param_req != ARG_PARAM_NONE)
			needs_set = true;
		if (a->arg_req != ARG_OPTIONAL && a->arg_req != ARG_HIDDEN)
			needs_set = true;
		if (a->validate_phase != ARG_CALLBACK_ALWAYS ||
		    a->action_phase != ARG_CALLBACK_ALWAYS)
			needs_set = true;
		if (a->_.deps || a->_.cons || a->_.subs)
			needs_set = true;
		if (needs_set)
			arg_set_new(a);
	}

	size_t ndeps = 0;
	size_t ncons = 0;
	size_t nsubs = 0;

	if (!a->_.deps) {
		if (a->_.deps_n > 0) {
			args_pd("%s has deps_n=%zu but deps=NULL", arg_str(a),
				a->_.deps_n);
			args_pd("Add dependencies using ARG_DEPENDS()");
			args_pi(a);
			args_abort();
		}

		if (a->_.deps_phase != ARG_RELATION_PARSE) {
			args_pd("%s has relation phase but no dependencies",
				arg_str(a));
			args_pi(a);
			args_abort();
		}

		goto arg_no_deps;
	}

	while (a->_.deps[ndeps])
		ndeps++;

	if (ndeps != a->_.deps_n) {
		args_pd("%s deps_n=%zu but actual is %zu", arg_str(a),
			a->_.deps_n, ndeps);
		args_pd("Add dependencies using ARG_DEPENDS()");
		args_pi(a);
		args_abort();
	}

	for_each_rel(a, deps, dep) {
		if (!dep) {
			args_pd("%s NULL deps[%zu]", arg_str(a), depi);
			args_pi(a);
			args_abort();
		}

		if (dep == a) {
			args_pd("%s depends on itself", arg_str(a));
			args_pi(a);
			args_abort();
		}

		if (!dep->set)
			arg_set_new(dep);
	}

arg_no_deps:
	if (!a->_.cons) {
		if (a->_.cons_n > 0) {
			args_pd("%s cons_n=%zu but cons=NULL", arg_str(a),
				a->_.cons_n);
			args_pd("Add conflicts using ARG_CONFLICTS()");
			args_pi(a);
			args_abort();
		}

		if (a->_.cons_phase != ARG_RELATION_PARSE) {
			args_pd("%s has relation phase but no conflicts",
				arg_str(a));
			args_pi(a);
			args_abort();
		}

		goto arg_no_cons;
	}

	while (a->_.cons[ncons])
		ncons++;

	if (ncons != a->_.cons_n) {
		args_pd("%s cons_n=%zu but actual is %zu", arg_str(a),
			a->_.cons_n, ncons);
		args_pd("Add conflicts using ARG_CONFLICTS()");
		args_pi(a);
		args_abort();
	}

	for_each_rel(a, cons, con) {
		if (!con) {
			args_pd("%s NULL cons[%zu]", arg_str(a), coni);
			args_pi(a);
			args_abort();
		}

		if (con == a) {
			args_pd("%s conflicts itself", arg_str(a));
			args_pi(a);
			args_abort();
		}

		if (!con->set)
			arg_set_new(con);

		for_each_rel(a, deps, dep) {
			if (dep != con)
				continue;

			args_pd("%s both depends and conflicts %s", arg_str(a),
				arg_str(con));
			args_pi(a);
			args_abort();
		}
	}

arg_no_cons:
	if (!a->_.subs) {
		if (a->_.subs_n > 0) {
			args_pd("%s subs_n=%zu but subs=NULL", arg_str(a),
				a->_.subs_n);
			args_pd("Specify subsets using ARG_SUBSETS()");
			args_pi(a);
			args_abort();
		}

		if (a->_.subs_strs) {
			args_pd("%s has subs_strs but no subsets", arg_str(a));
			args_pi(a);
			args_abort();
		}

		goto arg_no_subs;
	}

	while (a->_.subs[nsubs])
		nsubs++;

	if (nsubs != a->_.subs_n) {
		args_pd("%s subs_n=%zu but actual is %zu", arg_str(a),
			a->_.subs_n, nsubs);
		args_pd("Specify subset args using ARG_SUBSETS()");
		args_pi(a);
		args_abort();
	}

	if (a->_.subs_strs) {
		size_t nsstrs = 0;
		while (a->_.subs_strs[nsstrs])
			nsstrs++;

		if (nsstrs != a->_.subs_n) {
			args_pd("%s subs_n=%zu but subs_strs has %zu entries",
				arg_str(a), a->_.subs_n, nsstrs);
			args_pd("Both lists must be the same size");
			args_pi(a);
			args_abort();
		}
	}

	for_each_rel(a, subs, sub) {
		if (sub == a) {
			args_pd("%s subsets itself", arg_str(a));
			args_pi(a);
			args_abort();
		}

		if (!sub->set)
			arg_set_new(sub);

		if (a->param_req != ARG_PARAM_REQUIRED &&
		    sub->param_req == ARG_PARAM_REQUIRED &&
		    (!a->_.subs_strs || a->_.subs_strs[subi] == ARG_SUBPASS)) {
			args_pd("%s requires param but superset %s might not and has no custom string",
				arg_str(sub), arg_str(a));
			args_pi(a);
			args_abort();
		}

		if (!a->set)
			arg_set_new(a);

		for_each_rel(a, cons, con) {
			if (con == sub) {
				args_pd("%s both supersets and conflicts %s",
					arg_str(a), arg_str(sub));
				args_pi(a);
				args_abort();
			}
		}

		for_each_rel(sub, deps, dep) {
			if (dep == a) {
				args_pd("%s supersets %s but also depends on it",
					arg_str(a), arg_str(sub));
				args_pi(a);
				args_abort();
			}
		}
	}

arg_no_subs:
	for_each_arg(c, args) {
		if ((a->opt && c->opt && a->opt == c->opt) ||
		    (a->lopt && c->lopt && strcmp(a->lopt, c->lopt) == 0)) {
			args_pd("%s same opts as %s", arg_str(a), arg_str(c));
			args_pi(a);
			args_abort();
		}
	}

#define args_insert(list)                \
	do {                             \
		a->_.next_##list = list; \
		list = a;                \
	} while (0);

	args_insert(args);
	args_insert(help);
	args_insert(validate);
	args_insert(action);
#undef args_insert

	size_t len = ARGS_STR_PREPAD + strlen(arg_str(a));
	if (a->param)
		len += ARGS_PARAM_OFFSET + strlen(a->param);
	size_t check = len;
	if (a->help)
		check += ARGS_HELP_OFFSET + strlen(a->help);
	if (check >= ARG_STR_BUF_SIZE) {
		args_pd("%s combined opt, lopt, help string too long: %zu chars",
			arg_str(a), check);
		args_pi(a);
		args_abort();
	}
	a->_.help_len = len;
	if (len > longest)
		longest = len;

	a->_.valid = true;
	args_num++;
}

static bool arg_process(struct argument *a, const char *str)
{
	if (!a->_.valid) {
		args_pd("%s has internals pre-set", arg_str(a));
		args_pd("Please register arguments using ARGUMENT()");
		args_pi(a);
		args_abort();
	}

	if (a->set && *a->set) {
		args_pe("Argument %s specified multiple times", arg_str(a));
		return false;
	}

	if (a->parse_callback) {
		struct arg_callback ret = a->parse_callback(str, a->dest);
		if (ret.error) {
			args_pe("%s: %s", arg_str(a), ret.error);
			return false;
		}
	}

	if (a->_.deps_phase == ARG_RELATION_PARSE) {
		for_each_rel(a, deps, dep) {
			if (!*dep->set) {
				args_pe("%s requires %s to be set first",
					arg_str(a), arg_str(dep));
				return false;
			}
		}
	}

	if (a->_.cons_phase == ARG_RELATION_PARSE) {
		for_each_rel(a, cons, con) {
			if (*con->set) {
				args_pe("%s conflicts with %s", arg_str(a),
					arg_str(con));
				return false;
			}
		}
	}

	if (a->set)
		*a->set = true;

	for_each_rel(a, subs, sub) {
		if (*sub->set)
			continue;

		const char *sub_str = str;
		if (a->_.subs_strs && a->_.subs_strs[subi] &&
		    a->_.subs_strs[subi] != ARG_SUBPASS)
			sub_str = a->_.subs_strs[subi];

		if (!arg_process(sub, sub_str))
			return false;
	}

	return true;
}

static bool arg_parse_lopt(int *i)
{
	char *arg = argr.v[*i];
	char *name = arg + 2;
	char *value = strchr(name, '=');
	size_t name_len = value ? (size_t)(value - name) : strlen(name);
	if (value)
		value++;

	struct argument *a = NULL;
	for_each_arg(c, args) {
		if (c->lopt && strncmp(c->lopt, name, name_len) == 0 &&
		    c->lopt[name_len] == '\0') {
			a = c;
			break;
		}
	}

	if (!a) {
		args_pe("Unknown: --%.*s", (int)name_len, name);
		return false;
	}

	const char *str = NULL;
	if (a->param_req == ARG_PARAM_REQUIRED) {
		if (value) {
			str = value;
		} else if (*i + 1 < argr.c) {
			str = argr.v[++(*i)];
		} else {
			args_pe("--%s requires a parameter", a->lopt);
			return false;
		}
	} else if (a->param_req == ARG_PARAM_OPTIONAL) {
		if (value)
			str = value;
	} else {
		if (value) {
			args_pe("--%s does not take a parameter", a->lopt);
			return false;
		}
	}

	return arg_process(a, str);
}

static bool arg_parse_opt(int *i)
{
	char *arg = argr.v[*i];
	for (size_t j = 1; arg[j]; j++) {
		char opt = arg[j];

		struct argument *a = NULL;
		for_each_arg(c, args) {
			if (c->opt == opt) {
				a = c;
				break;
			}
		}

		if (!a) {
			args_pe("Unknown: -%c", opt);
			return false;
		}

		const char *str = NULL;
		if (a->param_req == ARG_PARAM_REQUIRED) {
			if (arg[j + 1]) {
				str = arg + j + 1;
				j = strlen(arg);
			} else if (*i + 1 < argr.c) {
				str = argr.v[++(*i)];
			} else {
				args_pe("-%c requires a parameter", opt);
				return false;
			}
		} else if (a->param_req == ARG_PARAM_OPTIONAL) {
			if (arg[j + 1]) {
				str = arg + j + 1;
				j = strlen(arg);
			}
		}

		if (!arg_process(a, str))
			return false;
	}
	return true;
}

#define args_vaorder(list)                                                   \
	do {                                                                 \
		for_each_arg(a, list) {                                      \
			struct argument *order = a->list##_order;            \
			if (order == NULL || order == (struct argument *)-1) \
				continue;                                    \
                                                                             \
			bool found = false;                                  \
			for_each_arg(check, list) {                          \
				if (check == order) {                        \
					found = true;                        \
					break;                               \
				}                                            \
			}                                                    \
                                                                             \
			if (!found) {                                        \
				args_pd("%s has invalid argument in " #list  \
					"_order",                            \
					arg_str(a));                         \
				args_pi(a);                                  \
				args_abort();                                \
			}                                                    \
		}                                                            \
	} while (0)

#define args_reorder(list)                                                             \
	do {                                                                           \
		struct argument *ordered = NULL;                                       \
		struct argument *unordered = list;                                     \
		list = NULL;                                                           \
                                                                                       \
		struct argument **pp = &unordered;                                     \
		while (*pp) {                                                          \
			struct argument *a = *pp;                                      \
			if (a->list##_order == (struct argument *)-1) {                \
				*pp = a->_.next_##list;                                \
				a->_.next_##list = ordered;                            \
				ordered = a;                                           \
			} else {                                                       \
				pp = &(*pp)->_.next_##list;                            \
			}                                                              \
		}                                                                      \
                                                                                       \
		bool changed = true;                                                   \
		while (unordered && changed) {                                         \
			changed = false;                                               \
			pp = &unordered;                                               \
			while (*pp) {                                                  \
				struct argument *a = *pp;                              \
				struct argument *ord = a->list##_order;                \
				bool can_place = false;                                \
				struct argument **insert_pos = NULL;                   \
                                                                                       \
				if (ord == NULL) {                                     \
					can_place = true;                              \
					if (!ordered) {                                \
						insert_pos = &ordered;                 \
					} else {                                       \
						struct argument *cur =                 \
							ordered;                       \
						while (cur->_.next_##list)             \
							cur = cur->_.next_##list;      \
						insert_pos =                           \
							&cur->_.next_##list;           \
					}                                              \
				} else {                                               \
					struct argument **pord = &ordered;             \
					while (*pord) {                                \
						if (*pord == ord) {                    \
							can_place = true;              \
							insert_pos =                   \
								&(*pord)->_            \
									 .next_##list; \
							break;                         \
						}                                      \
						pord = &(*pord)->_.next_##list;        \
					}                                              \
				}                                                      \
                                                                                       \
				if (can_place && insert_pos) {                         \
					*pp = a->_.next_##list;                        \
					a->_.next_##list = *insert_pos;                \
					*insert_pos = a;                               \
					changed = true;                                \
				} else {                                               \
					pp = &(*pp)->_.next_##list;                    \
				}                                                      \
			}                                                              \
		}                                                                      \
                                                                                       \
		if (unordered) {                                                       \
			if (!ordered) {                                                \
				ordered = unordered;                                   \
			} else {                                                       \
				struct argument *cur = ordered;                        \
				while (cur->_.next_##list)                             \
					cur = cur->_.next_##list;                      \
				cur->_.next_##list = unordered;                        \
			}                                                              \
		}                                                                      \
                                                                                       \
		list = ordered;                                                        \
	} while (0)

bool args_parse(int argc, char *argv[])
{
	argr.c = argc;
	argr.v = argv;

	args_vaorder(help);
	args_vaorder(validate);
	args_vaorder(action);
#undef args_vaorder

	args_reorder(help);
	args_reorder(validate);
	args_reorder(action);
#undef args_reorder

	bool success = true;

	for (int i = 1; i < argr.c; i++) {
		char *arg = argr.v[i];

		if (strcmp(arg, "--") == 0)
			break;

		if (arg[0] != '-')
			continue;

		if (arg[1] == '\0')
			continue;

		if (arg[1] == '-') {
			if (!arg_parse_lopt(&i))
				success = false;
		} else {
			if (!arg_parse_opt(&i))
				success = false;
		}
	}

	return success;
}

bool args_validate(void)
{
	bool any_invalid = false;

	for_each_arg(a, validate) {
		if (!a->_.valid) {
			args_pd("%s has internals pre-set", arg_str(a));
			args_pd("Please register arguments using ARGUMENT()");
			args_pi(a);
			args_abort();
		}

		if (a->arg_req == ARG_REQUIRED && !*a->set) {
			bool any_conflict_set = false;
			for_each_rel(a, cons, con) {
				if (*con->set) {
					any_conflict_set = true;
					break;
				}
			}
			if (!any_conflict_set) {
				args_pe("Missing required argument: %s",
					arg_str(a));
				a->_.valid = false;
				any_invalid = true;
			}
		}

		bool should_check_deps = false;
		switch (a->_.deps_phase) {
		case ARG_RELATION_PARSE:
			break;
		case ARG_RELATION_VALIDATE_ALWAYS:
			should_check_deps = true;
			break;
		case ARG_RELATION_VALIDATE_SET:
			should_check_deps = *a->set;
			break;
		case ARG_RELATION_VALIDATE_UNSET:
			should_check_deps = !*a->set;
			break;
		default:
			args_pd("Unknown dependency relation phase in %s",
				arg_str(a));
			args_pi(a);
			break;
		}

		if (should_check_deps) {
			for_each_rel(a, deps, dep) {
				if (*dep->set)
					continue;
				args_pe("%s requires %s to be set", arg_str(a),
					arg_str(dep));
				any_invalid = true;
			}
		}

		bool should_check_cons = false;
		switch (a->_.cons_phase) {
		case ARG_RELATION_PARSE:
			break;
		case ARG_RELATION_VALIDATE_ALWAYS:
			should_check_cons = true;
			break;
		case ARG_RELATION_VALIDATE_SET:
			should_check_cons = *a->set;
			break;
		case ARG_RELATION_VALIDATE_UNSET:
			should_check_cons = !*a->set;
			break;
		default:
			args_pd("Unknown conflict relation phase in %s",
				arg_str(a));
			args_pi(a);
			break;
		}

		if (should_check_cons) {
			for_each_rel(a, cons, con) {
				if (!*con->set)
					continue;
				args_pe("%s conflicts with %s", arg_str(a),
					arg_str(con));
				any_invalid = true;
			}
		}

		if (!a->validate_callback)
			continue;

		bool should_validate = false;
		switch (a->validate_phase) {
		case ARG_CALLBACK_ALWAYS:
			should_validate = true;
			break;
		case ARG_CALLBACK_IF_SET:
			should_validate = *a->set;
			break;
		case ARG_CALLBACK_IF_UNSET:
			should_validate = !*a->set;
			break;
		default:
			args_pd("Unknown .validate enum in %s", arg_str(a));
			args_pi(a);
			args_abort();
		}

		if (should_validate) {
			struct arg_callback ret = a->validate_callback();
			if (ret.error) {
				args_pe("%s: %s", arg_str(a), ret.error);
				any_invalid = true;
				a->_.valid = false;
			}
		}
	}

	return !any_invalid;
}

void args_actions(void)
{
	for_each_arg(a, action) {
		if (!a->action_callback)
			continue;

		bool should_act = false;
		switch (a->action_phase) {
		case ARG_CALLBACK_ALWAYS:
			should_act = true;
			break;
		case ARG_CALLBACK_IF_SET:
			should_act = *a->set;
			break;
		case ARG_CALLBACK_IF_UNSET:
			should_act = !*a->set;
			break;
		default:
			args_pd("Unknown .action enum in %s", arg_str(a));
			args_pi(a);
			args_abort();
		}

		if (should_act)
			a->action_callback();
	}
}

static void arg_print_help(const struct argument *a)
{
	printf("%*s%s", ARGS_STR_PREPAD, "", arg_str(a));
	if (a->param)
		printf("%*s%s", ARGS_PARAM_OFFSET, "", a->param);

	if (!a->help) {
		printf("\n");
		return;
	}

	size_t off = longest + ARGS_HELP_OFFSET;
	size_t pad = off > a->_.help_len ? off - a->_.help_len : 1;

	bool first = true;
	const char *phelp = a->help;
	while (*phelp) {
		const char *nl = strchr(phelp, '\n');
		size_t line = nl ? (size_t)(nl - phelp) : strlen(phelp);

		if (first) {
			printf("%*s%.*s", (int)pad, "", (int)line, phelp);
			first = false;
		} else {
			printf("\n%*s%.*s", (int)off, "", (int)line, phelp);
		}

		if (!nl)
			break;
		phelp = nl + 1;
	}

	printf("\n");
}

#ifdef ARGS_NO_DEFAULT_HELP
void args_print_help(void)
#else
static void args_print_help(void)
#endif
{
	printf("Usage: %s [ARGUMENTS]\n", argr.v[0]);

	bool first = true;
	for_each_arg(a, help) {
		if (a->arg_req == ARG_OPTIONAL || a->arg_req == ARG_HIDDEN)
			continue;
		if (first) {
			printf("\nRequired arguments:\n");
			first = false;
		}
		arg_print_help(a);
	}

	first = true;
	for_each_arg(a, help) {
		if (a->arg_req != ARG_OPTIONAL || a->arg_req == ARG_HIDDEN)
			continue;
		if (first) {
			printf("\nOptional arguments:\n");
			first = false;
		}
		arg_print_help(a);
	}
}

#ifndef ARGS_NO_DEFAULT_HELP
static struct arg_callback print_help(const char *str, void *dest)
{
	(void)str;
	(void)dest;
	args_print_help();
	exit(EXIT_SUCCESS);
}

ARGUMENT(help) = {
	.parse_callback = print_help,
	.help = "Display this help message",
	.lopt = "help",
	.opt = 'h',
};
#endif

#undef for_each_arg
#undef for_each_rel

#else

#endif /* ARGS_IMPLEMENTATION */

#endif /* UTIL_ARGS_H */
