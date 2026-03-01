#pragma once
#ifndef UTIL_ARGS_H
#define UTIL_ARGS_H

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>

/**
 * @file args.h
 * @author Jakov Dragičević <jakodrag345@gmail.com> (GitHub: jakovdev)
 * @brief Header-only, decentralized CLI argument framework for C.
 * @details
 * This library lets each translation unit declare and register arguments near
 * the code that owns them, instead of maintaining a single global argument
 * table. Registration is constructor-driven and therefore complete before
 * calling @ref args_parse.
 *
 * Correctness depends on constructor attribute support (GCC/Clang) or 
 * .CRT$XCU sections (MSVC) executing registration hooks before @c main.
 *
 * @note Partial C++ support due to struct field declaration order.
 * Most macros will break because they don't take this into account.
 * You can manually initialize fields which should make it work like in C.
 *
 * @see @ref args_quick_start "Quick start"
 */

/**
 * @mainpage Decentralized CLI Argument Framework
 *
 * @section args_intro Overview
 * @ref args.h "args.h" provides a header-only argument manager intended for
 * modular C codebases. Arguments are declared with @ref ARGUMENT in any source
 * file, registered automatically, then processed in three explicit phases:
 * parsing, validation, and actions.
 *
 * @section args_features Feature Highlights
 * - Distributed argument declarations with constructor-based registration.
 * - Automatic help generation.
 * - Dependency/conflict/subset relationships between arguments.
 * - Automatic validation of required arguments and relation rules.
 * - Typed parsing helpers via @ref ARG_PARSER family.
 * - Validation and general callbacks with flexible execution policies.
 * - Deterministic cross-file ordering for help, validate, and action phases.
 *
 * @section args_quick_start Quick Start
 * @code{.c}
 * #define ARGS_IMPLEMENTATION
 * #include "args.h"
 *
 * static bool verbose;
 * ARGUMENT(verbose) = {
 *     .opt = 'v',
 *     .lopt = "verbose",
 *     .help = "Enable verbose output",
 *     .set = &verbose,
 * };
 *
 * int main(int argc, char **argv) {
 *     if (!args_parse(argc, argv) || !args_validate())
 *         return 1;
 *     args_actions(); // Optional
 *     return 0;
 * }
 * @endcode
 *
 * @section args_navigation Docs Navigation
 * - @ref args_core
 * - @ref args_macros
 * - @ref args_relations
 * - @ref args_callbacks
 * - @ref args_types
 * - @ref args_parsers
 * - @ref args_customizable
 * - @ref args_internals
 */

/** @defgroup args_core Core API */
/** @defgroup args_macros Definition Macros */
/** @defgroup args_relations Relationships */
/** @defgroup args_callbacks Callback Phases */
/** @defgroup args_types Data Types */
/** @defgroup args_parsers Parser Helpers */
/** @defgroup args_customizable Customization Points */
/** @defgroup args_internals Internals */

/** @name Creating New Arguments */
/** @{ */

#ifndef __cplusplus
/** @ingroup args_macros
 * @def ARGUMENT
 * @brief Creates and registers a new argument object.
 * @hideinitializer
 * @details
 * Use this macro at file scope. It declares @c struct argument
 * @c _arg_<name>, wires constructor-based registration, then leaves a second
 * declaration target for designated initialization.
 *
 * @param name Unique argument identifier used in generated symbol names.
 * @pre The identifier is unique within the link unit.
 * @post The argument is linked into internal processing lists before @c main.
 * @see ARG_DECLARE
 * @see ARG_EXTERN
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

/** @} */

/** @name Requirement And Visibility */
/** @{ */

/** @ingroup args_types
 * @enum arg_requirement
 * @brief Requirement policy for @ref argument::arg_req.
 */
enum arg_requirement {
	ARG_OPTIONAL, /**< User does not need to provide the argument. */
	ARG_REQUIRED, /**< Argument must be present unless conflicted out. */
	ARG_HIDDEN, /**< Optional and omitted from generated help output. */
	ARG_SOMETIME, /**< Conditionally required, enforce via relations/callbacks. */
};

/** @} */

/** @name Parameter Modes */
/** @{ */

/** @ingroup args_types
 * @enum arg_parameter
 * @brief Parameter requirement for @ref argument::param_req.
 * @remark Semantics intentionally match common @c getopt usage.
 */
enum arg_parameter {
	ARG_PARAM_NONE, /**< Argument is a flag, does not take a parameter. */
	ARG_PARAM_REQUIRED, /**< Callback receives non-NULL @p str. */
	ARG_PARAM_OPTIONAL, /**< Callback may receive NULL @p str. */
};

/** @} */

/** @addtogroup args_parsers
 * @par Built-in parser wrappers
 * @ref ARG_PARSE_L, @ref ARG_PARSE_LL, @ref ARG_PARSE_UL,
 * @ref ARG_PARSE_ULL, @ref ARG_PARSE_F, and @ref ARG_PARSE_D are convenience
 * wrappers around @ref ARG_PARSER.
 */

/** @ingroup args_parsers
 * @def ARG_PARSER
 * @brief Generates a typed parser callback with common conversion checks.
 * @hideinitializer
 * @details
 * The generated function is named @c parse_<name> and matches @ref argument::parse_callback.
 * It parses @p str, rejects malformed input, range errors, and custom condition failures,
 * then writes to @p dest.
 *
 * @param name Suffix used for generated function name.
 * @param strto Conversion routine, e.g. @c strtol or @c strtod.
 * @param ARG_BASE Macro, leave out macro for no base, or e.g. ARG_BASE(10) to pass to strto.
 * @param strto_t Return type of @p strto.
 * @param dest_t Destination pointee type.
 * @param CAST Optional cast expression when narrowing/widening.
 * @param cond Rejection predicate evaluated against local variable @c val.
 * @param err Error text, falls back to @ref ARG_ERR when empty.
 * @retval ARG_VALID Input accepted and stored in @p dest.
 * @retval ARG_INVALID Input rejected with diagnostic message.
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

/** @ingroup args_parsers
 * @def ARG_BASE
 * @brief Helper for the @p ARG_BASE parameter in @ref ARG_PARSER.
 * @hideinitializer
 */
#define ARG_BASE(N) , N

#ifndef ARG_ERR
/** @ingroup args_parsers
 * @def ARG_ERR
 * @brief Default parser error message if @p err is empty in @ref ARG_PARSER.
 * @remark Overridable before including @ref args.h.
 */
#define ARG_ERR "Invalid value"
#endif

/** @ingroup args_parsers
 * @def ARG_PARSE_L
 * @brief Convenience wrapper of @ref ARG_PARSER using @c strtol.
 * @hideinitializer
 */
#define ARG_PARSE_L(name, base, dest_t, CAST, cond, err) \
	ARG_PARSER(name, strtol, ARG_BASE(base), long, dest_t, CAST, cond, err)

/** @ingroup args_parsers
 * @def ARG_PARSE_LL
 * @brief Convenience wrapper of @ref ARG_PARSER using @c strtoll.
 * @hideinitializer
 */
#define ARG_PARSE_LL(name, base, dest_t, CAST, cond, err)                  \
	ARG_PARSER(name, strtoll, ARG_BASE(base), long long, dest_t, CAST, \
		   cond, err)

/** @ingroup args_parsers
 * @def ARG_PARSE_UL
 * @brief Convenience wrapper of @ref ARG_PARSER using @c strtoul.
 * @hideinitializer
 */
#define ARG_PARSE_UL(name, base, dest_t, CAST, cond, err)                      \
	ARG_PARSER(name, strtoul, ARG_BASE(base), unsigned long, dest_t, CAST, \
		   cond, err)

/** @ingroup args_parsers
 * @def ARG_PARSE_ULL
 * @brief Convenience wrapper of @ref ARG_PARSER using @c strtoull.
 * @hideinitializer
 */
#define ARG_PARSE_ULL(name, base, dest_t, CAST, cond, err)                     \
	ARG_PARSER(name, strtoull, ARG_BASE(base), unsigned long long, dest_t, \
		   CAST, cond, err)

/** @ingroup args_parsers
 * @def ARG_PARSE_F
 * @brief Convenience wrapper of @ref ARG_PARSER using @c strtof.
 * @hideinitializer
 */
#define ARG_PARSE_F(name, dest_t, CAST, cond, err) \
	ARG_PARSER(name, strtof, , float, dest_t, CAST, cond, err)

/** @ingroup args_parsers
 * @def ARG_PARSE_D
 * @brief Convenience wrapper of @ref ARG_PARSER using @c strtod.
 * @hideinitializer
 */
#define ARG_PARSE_D(name, dest_t, CAST, cond, err) \
	ARG_PARSER(name, strtod, , double, dest_t, CAST, cond, err)

/** @addtogroup args_parsers
 * @par Custom parser contract
 * Custom parser callbacks return @ref ARG_INVALID or @ref ARG_VALID and
 * use signature @c (const char *str, void *dest).
 */

/** @ingroup args_types
 * @struct arg_callback
 * @brief Result object returned by parser/validator callbacks.
 * @invariant @ref arg_callback::error is @c NULL when the callback succeeds.
 */
struct arg_callback {
	/** @var arg_callback::error
	 * @brief Diagnostic user-facing message for failures, NULL on success.
	 */
	const char *error;
};

/** @ingroup args_callbacks
 * @def ARG_INVALID
 * @brief Creates a failing @ref arg_callback value.
 */
#define ARG_INVALID(msg) ((struct arg_callback){ .error = msg })

/** @ingroup args_callbacks
 * @def ARG_VALID
 * @brief Creates a successful @ref arg_callback value.
 */
#define ARG_VALID() ((struct arg_callback){ .error = NULL })

/** @ingroup args_types
 * @struct args_raw
 * @brief Copy of the raw process argument vector.
 * @details Exposed for use cases such as rendering usage with executable name.
 */
struct args_raw {
	int c; /**< Argument count. */
	char **v; /**< Argument vector. */
};

#ifdef ARGS_GLOBAL_ARGR
/** @ingroup args_customizable
 * @var argr
 * @brief Global storage of raw @c argc/@c argv values.
 * @note Available when @c ARGS_GLOBAL_ARGR is defined by the including file.
 */
extern struct args_raw argr;
#endif

/** @ingroup args_core
 * @brief Parses command line arguments and runs parse-phase relations.
 * @details
 * For each user-provided option, the parser resolves the matching @ref argument
 * and calls @ref argument::parse_callback when available.
 * Parse-time dependency/conflict relations are enforced immediately.
 * Duplicate-argument detection applies only when @ref argument::set is non-NULL,
 * though keep in mind it is implicitly allocated for most features.
 *
 * @par Execution order
 * For each user-provided argument:
 * -# If argument takes a parameter and is already set, error for repeated argument
 * -# Run @ref argument::parse_callback if it exists
 * -# Check dependency/conflict relations if @ref arg_relation_phase is @ref ARG_RELATION_PARSE
 * -# Set @ref argument::set to true (implicit allocation if needed)
 *
 * @pre Every argument was declared via @ref ARGUMENT and linked in.
 * @post All successfully processed arguments with non-NULL @ref argument::set
 * have @c *set == true.
 * @retval true Parse succeeded for all encountered options.
 * @retval false At least one user-facing parse error occurred.
 * @see args_validate
 */
bool args_parse(int argc, char *argv[]);

/** @name Callback Phases */
/** @{ */

/** @addtogroup args_callbacks
 * @par Lifecycle
 * Execution proceeds through parse, validate, and action stages. Internal
 * schema checks happen inside library logic. Project-specific checks belong in
 * your callbacks.
 */

/** @ingroup args_callbacks
 * @enum arg_callback_phase
 * @brief Execution policy for @ref argument::validate_phase and @ref argument::action_phase.
 */
enum arg_callback_phase {
	ARG_CALLBACK_ALWAYS, /**< Run regardless of user input state. */
	ARG_CALLBACK_IF_SET, /**< Run only when the argument was provided. */
	ARG_CALLBACK_IF_UNSET, /**< Run only when the argument was not provided. */
};

/** @} */

/** @ingroup args_core
 * @brief Validates argument schema rules and user-provided combinations.
 * @details
 * This stage enforces required-argument rules, non-parse relation phases, and
 * executes @ref argument::validate_callback according to @ref argument::validate_phase.
 *
 * @par Validation checks
 * For each registered argument:
 * -# If @ref ARG_REQUIRED, check it was provided (unless conflicted out)
 * -# Check dependency/conflict relations if @ref arg_relation_phase is not @ref ARG_RELATION_PARSE
 * -# Run @ref argument::validate_callback if it exists using @ref argument::validate_phase
 *
 * @pre @ref args_parse was called.
 * @post All validation callbacks due in this pass have executed.
 * @retval true All checks passed.
 * @retval false One or more constraints failed.
 */
bool args_validate(void);

/** @ingroup args_core
 * @brief Executes action callbacks for arguments matching action phase rules.
 * @pre @ref args_validate returned @c true.
 * @post All eligible @ref argument::action_callback functions have run.
 */
void args_actions(void);

/** @addtogroup args_relations
 * @brief Dependencies, conflicts, and subset propagation.
 * @{ */

/**
 * @details
 * Relations model command schemas declaratively:
 * - dependency: argument requires others
 * - conflict: argument excludes others
 * - subset: setting an argument that lists subset arguments triggers them
 */

/** @ingroup args_relations
 * @def ARG
 * @brief Returns the address of argument object @p name.
 * @hideinitializer
 */
#define ARG(name) &_arg_##name

/** @ingroup args_macros
 * @def ARG_DECLARE
 * @brief Forward declares an argument for same-file forward references.
 * @hideinitializer
 * @details
 * Example:
 * @code{.c}
 * ARG_DECLARE(foo);
 * ARGUMENT(bar) = {
 *     ...
 *     ARG_DEPENDS(ARG_RELATION_PARSE, ARG(foo)),
 *     ...
 * };
 * ARGUMENT(foo) = { ... };
 * @endcode
 */
#define ARG_DECLARE(name) struct argument _arg_##name

/** @ingroup args_macros
 * @def ARG_EXTERN
 * @brief Extern declaration for an argument defined in another file.
 * @hideinitializer
 * @details
 * Example:
 *
 * foo.c
 * @code{.c}
 * ARGUMENT(foo) = { ... };
 * @endcode
 *
 * bar.c
 * @code{.c}
 * ARG_EXTERN(foo);
 * ARGUMENT(bar) = {
 *     ...
 *     ARG_DEPENDS(ARG_RELATION_PARSE, ARG(foo)),
 *     ...
 * };
 * @endcode
 */
#define ARG_EXTERN(name) extern struct argument _arg_##name

/** @ingroup args_relations
 * @def ARG_DEPENDS
 * @brief Declares dependency relations for an argument.
 * @hideinitializer
 * @details
 * When setting this argument, all dependencies must already be set.
 * Use inside an @ref ARGUMENT definition.
 *
 * Example:
 * @code{.c}
 * ARGUMENT(foo) = { ... };
 * ARGUMENT(bar) = { ... };
 * ARGUMENT(baz) = {
 *     ...
 *     ARG_DEPENDS(ARG_RELATION_PARSE, ARG(foo), ARG(bar)),
 *     ...
 * };
 * @endcode
 *
 * @param relation_phase When to check dependencies, see @ref arg_relation_phase.
 */
#define ARG_DEPENDS(relation_phase, ...)                           \
	._.deps_phase = relation_phase,                            \
	._.deps = (struct argument *[]){ __VA_ARGS__, NULL },      \
	._.deps_n = sizeof((struct argument *[]){ __VA_ARGS__ }) / \
		    sizeof(struct argument *)

/** @ingroup args_relations
 * @def ARG_CONFLICTS
 * @brief Declares conflict relations for an argument.
 * @hideinitializer
 * @details
 * If a conflict argument is set, this one must not be set, overriding
 * ARG_REQUIRED. Use inside an @ref ARGUMENT definition.
 *
 * Example:
 * @code{.c}
 * ARGUMENT(foo) = { ... };
 * ARGUMENT(bar) = { ... };
 * ARGUMENT(baz) = {
 *     ...
 *     ARG_CONFLICTS(ARG_RELATION_PARSE, ARG(foo), ARG(bar)),
 *     ...
 * };
 * @endcode
 *
 * @param relation_phase When to check conflicts, see @ref arg_relation_phase.
 */
#define ARG_CONFLICTS(relation_phase, ...)                         \
	._.cons_phase = relation_phase,                            \
	._.cons = (struct argument *[]){ __VA_ARGS__, NULL },      \
	._.cons_n = sizeof((struct argument *[]){ __VA_ARGS__ }) / \
		    sizeof(struct argument *)

/** @ingroup args_relations
 * @def ARG_SUBSETS
 * @brief Declares subset arguments triggered by a parent argument.
 * @hideinitializer
 * @details
 * When this argument is set, all subsets are also processed. Parent string
 * is passed to subset parsers unless customized with @ref ARG_SUBSTRINGS.
 * Use inside an @ref ARGUMENT definition.
 *
 * Example:
 * @code{.c}
 * ARGUMENT(foo) = { ... };
 * ARGUMENT(bar) = { ... };
 * ARGUMENT(baz) = {
 *     ...
 *     ARG_SUBSETS(ARG(foo), ARG(bar)),
 *     ARG_SUBSTRINGS("out.txt", ARG_SUBPASS),
 *     ...
 * };
 * @endcode
 */
#define ARG_SUBSETS(...)                                           \
	._.subs = (struct argument *[]){ __VA_ARGS__, NULL },      \
	._.subs_n = sizeof((struct argument *[]){ __VA_ARGS__ }) / \
		    sizeof(struct argument *)

/** @ingroup args_relations
 * @def ARG_SUBSTRINGS
 * @brief Supplies index-aligned custom strings for subset processing.
 * @hideinitializer
 * @details
 * Use @ref ARG_SUBPASS to pass parent string, omit entirely if not needed.
 */
#define ARG_SUBSTRINGS(...) \
	._.subs_strs = ((const char *[]){ __VA_ARGS__, NULL })

/** @ingroup args_relations
 * @def ARG_SUBPASS
 * @brief Special marker meaning "forward parent string to subset parser".
 * @hideinitializer
 * @details
 * Use in @ref ARG_SUBSTRINGS to indicate that the parent argument's string
 * should be passed to the subset parser instead of a custom string.
 */
#define ARG_SUBPASS ((const char *)-1)

/** @ingroup args_relations
 * @enum arg_relation_phase
 * @brief Relation evaluation phase used by @ref ARG_DEPENDS and @ref ARG_CONFLICTS.
 */
enum arg_relation_phase {
	ARG_RELATION_PARSE, /**< Parse-time checks, effectively no-op when unset. */
	ARG_RELATION_VALIDATE_ALWAYS, /**< Validate-time checks regardless of argument state. */
	ARG_RELATION_VALIDATE_SET, /**< Validate-time checks only when the argument is set. */
	ARG_RELATION_VALIDATE_UNSET, /**< Validate-time checks only when the argument is not set. */
};

/** @} */

/** @addtogroup args_macros
 * @name Cross-File Execution Ordering
 * @brief Deterministic ordering controls for help/validate/action lists.
 * @{ */

/** @ingroup args_macros
 * @def ARG_ORDER_FIRST
 * @brief Place argument first in the selected ordered list.
 * @hideinitializer
 * @details
 * Value for @ref argument::validate_order, @ref argument::action_order, or @ref argument::help_order.
 */
#define ARG_ORDER_FIRST ((struct argument *)-1)

/** @ingroup args_macros
 * @def ARG_ORDER_AFTER
 * @brief Place argument immediately after another argument.
 * @hideinitializer
 * @details
 * Value for @ref argument::validate_order, @ref argument::action_order, or @ref argument::help_order.
 * Use @ref ARG_DECLARE or @ref ARG_EXTERN to forward-declare the referenced arg.
 */
#define ARG_ORDER_AFTER(arg_name) (ARG(arg_name))

/** @} */

/** @name Internals */
/** @{ */
/** @ingroup args_internals
 * @struct args_internal
 * @brief Internal runtime/link-list and relation metadata.
 */
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
/** @} */

/** @ingroup args_core
 * @ingroup args_types
 * @struct argument
 * @brief Declarative schema object for one CLI argument.
 * @details
 * Initialize this type through @ref ARGUMENT.
 * C designated initializers let you specify only the fields you need.
 * @invariant At least one of @ref argument::opt or @ref argument::lopt is set.
 */
struct argument {
	/** @name Storage */
	/** @{ */
	/** @var argument::set
	 * @brief Tracks whether the user supplied this argument.
	 * @details
	 * If NULL, it is allocated implicitly when needed by schema features.
	 */
	bool *set;

	/** @var argument::dest
	 * @brief Destination pointer passed as @p dest to parser callbacks.
	 */
	void *dest;
	/** @} */

	/** @name Callbacks */
	/** @{ */

	/** @var argument::parse_callback
	 * @brief Function invoked during @ref args_parse.
	 * @details
	 * Use @ref ARG_PARSER helpers for numeric parsing, or provide custom logic.
	 * Callbacks aren't limited to parsing e.g. --help exiting immediately.
	 * @note @ref argument::dest can be NULL
	 *
	 * Example:
	 * @code{.c}
	 * static double mydouble;
	 * ARG_PARSE_D(mydoubles, double, , val < 5.0, "Must be >= 5.0");
	 * ARGUMENT(myarg) = {
	 *     ...
	 *     .dest = &mydouble,
	 *     .parse_callback = parse_mydoubles,
	 *     ...
	 * };
	 * @endcode
	 *
	 * You can also write custom parsers:
	 * @code{.c}
	 * enum Color { COLOR_INVALID = -1, RED, GREEN, BLUE };
	 * static enum Color color = COLOR_INVALID;
	 * static struct arg_callback parse_color(const char *str, void *dest) {
	 *     // Using 'color' directly is also possible in this example
	 *     enum Color col = COLOR_INVALID;
	 *     if (strcmp(str, "red") == 0)
	 *         col = RED;
	 *     else if (strcmp(str, "green") == 0)
	 *         col = GREEN;
	 *     else if (strcmp(str, "blue") == 0)
	 *         col = BLUE;
	 *     else
	 *         return ARG_INVALID("Invalid color");
	 *     *(enum Color *)dest = col;
	 *     return ARG_VALID();
	 * }
	 * ARGUMENT(color) = {
	 *     ...
	 *     .dest = &color,
	 *     .parse_callback = parse_color,
	 *     ...
	 * };
	 * @endcode
	 *
	 * @param str User-provided value string.
	 * @param dest Alias of @ref argument::dest, potentially NULL.
	 * @retval ARG_VALID Callback accepted input.
	 * @retval ARG_INVALID Callback rejected input.
	 */
	struct arg_callback (*parse_callback)(const char *str, void *dest);

	/** @var argument::validate_callback
	 * @brief Function invoked during @ref args_validate.
	 * @details
	 * Use for cross-field or project-specific constraints after parse.
	 *
	 * Example:
	 * @code{.c}
	 * enum Color { RED, GREEN, BLUE };
	 * static enum Color color;
	 * static bool other_option;
	 * static struct arg_callback validate_color(void) {
	 *     if (color == RED && other_option)
	 *         return ARG_INVALID("Red color cannot be used.");
	 *     return ARG_VALID();
	 * }
	 * ARGUMENT(color) = {
	 *     ...
	 *     .dest = &color,
	 *     .validate_callback = validate_color,
	 *     ...
	 * };
	 * @endcode
	 *
	 * @retval ARG_VALID Validation succeeded.
	 * @retval ARG_INVALID Validation failed.
	 */
	struct arg_callback (*validate_callback)(void);

	/** @var argument::action_callback
	 * @brief Function invoked during @ref args_actions.
	 * @note Intended for side effects such as applying runtime configuration.
	 *
	 * Example:
	 * @code{.c}
	 * static bool verbose;
	 * static void print_config(void) {
	 *     printf("Verbose mode is on");
	 * }
	 * ARGUMENT(verbose) = {
	 *     ...
	 *     .set = &verbose,
	 *     .action_callback = print_config,
	 *     .action_phase = ARG_CALLBACK_IF_SET,
	 *     ...
	 * };
	 * @endcode
	 */
	void (*action_callback)(void);
	/** @} */

	/** @name Schema */
	/** @{ */

	/** @var argument::arg_req
	 * @brief Specifies whether this argument is required, optional, or hidden.
	 * @see arg_requirement
	 */
	enum arg_requirement arg_req;

	/** @var argument::param_req
	 * @brief Specifies whether this argument takes a parameter and if it's required.
	 * @see arg_parameter
	 */
	enum arg_parameter param_req;

	/** @var argument::validate_phase
	 * @brief Controls when @ref argument::validate_callback runs.
	 * @see arg_callback_phase
	 */
	enum arg_callback_phase validate_phase;

	/** @var argument::action_phase
	 * @brief Controls when @ref argument::action_callback runs.
	 * @see arg_callback_phase
	 */
	enum arg_callback_phase action_phase;
	/** @} */

	/** @name Ordering */
	/** @{ */

	/** @var argument::validate_order
	 * @brief Ordering for validation.
	 *
	 * Use @ref ARG_ORDER_FIRST or @ref ARG_ORDER_AFTER, NULL means append at the end.
	 */
	struct argument *validate_order;

	/** @var argument::action_order
	 * @brief Ordering for action execution.
	 *
	 * Use @ref ARG_ORDER_FIRST or @ref ARG_ORDER_AFTER, NULL means append at the end.
	 */
	struct argument *action_order;

	/** @var argument::help_order
	 * @brief Ordering for help display.
	 *
	 * Use @ref ARG_ORDER_FIRST or @ref ARG_ORDER_AFTER, NULL means append at the end.
	 */
	struct argument *help_order;
	/** @} */

	/** @name Help Presentation */
	/** @{ */

	/** @var argument::help
	 * @brief Help description for this argument.
	 *
	 * Multiline strings are supported (e.g. @c "Line1\nLine2").
	 * @see ARGS_STR_PREPAD
	 * @see ARGS_PARAM_OFFSET
	 * @see ARGS_HELP_OFFSET
	 */
	const char *help;

	/** @var argument::param
	 * @brief Parameter name for help display (e.g., "N" for a number).
	 *
	 * @invariant Required when @ref argument::param_req is not @ref ARG_PARAM_NONE.
	 */
	const char *param;

	/** @var argument::lopt
	 * @brief Long option name (e.g. @c "output" -> @c --output).
	 */
	const char *lopt;

	/** @var argument::opt
	 * @brief Short option character (e.g. @c 'o' -> @c -o).
	 */
	char opt;
	/** @} */

	/** @name Internals */
	/** @{ */
	/** @var argument::_
	 * @brief Internal runtime metadata populated by registration.
	 */
	struct args_internal _;
	/** @} */
};

#ifdef ARGS_NO_DEFAULT_HELP
/** @ingroup args_customizable
 * @brief Prints generated CLI help output.
 * @note Exposed only when @c ARGS_NO_DEFAULT_HELP is defined, enabling custom help
 * argument declarations by the including project.
 */
void args_print_help(void);
#endif

/** @name Internals */
/** @{ */
/** @ingroup args_internals
 * @brief Internal registration entry point used by @ref ARGUMENT.
 * @warning Prefer @ref ARGUMENT for all public usage.
 */
void _args_register(struct argument *);

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
/** @ingroup args_internals
 * @def _ARGS_CONSTRUCTOR
 * @brief Constructor abstraction for GCC/Clang/MSVC registration hooks.
 * @hideinitializer
 * @see https://stackoverflow.com/questions/1113409/attribute-constructor-equivalent-in-vc
 */
#define _ARGS_CONSTRUCTOR(f)                              \
	static void f(void) __attribute__((constructor)); \
	static void f(void)
#endif
/** @} */

#ifdef ARGS_IMPLEMENTATION

#ifndef args_pe
/** @ingroup args_customizable
 * @brief Error print.
 * @remark Overridable before including @ref args.h.
 */
#define args_pe(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef args_pd
/** @ingroup args_customizable
 * @brief Developer-only debug print.
 * @remark Overridable before including @ref args.h.
 */
#define args_pd(...) fprintf(stderr, __VA_ARGS__)
#endif

#ifndef args_pi
/** @ingroup args_customizable
 * @brief Internal error print, user-facing dev print.
 * @remark Overridable before including @ref args.h.
 */
#define args_pi(arg) args_pe("Internal error for %s", arg_str(arg))
#endif

#ifndef args_abort
/** @ingroup args_customizable
 * @brief Abort function.
 * @remark Overridable before including @ref args.h.
 */
#define args_abort() abort()
#endif

#ifndef ARGS_STR_PREPAD
/** @ingroup args_customizable
 * @brief Pre-padding for help text.
 * @remark Overridable before including @ref args.h.
 */
#define ARGS_STR_PREPAD (2)
#elif ARGS_STR_PREPAD < 0
#error ARGS_STR_PREPAD cannot be negative
#endif

#ifndef ARGS_PARAM_OFFSET
/** @ingroup args_customizable
 * @brief Offset between argument and parameter in help text.
 * @remark Overridable before including @ref args.h.
 */
#define ARGS_PARAM_OFFSET (1)
#elif ARGS_PARAM_OFFSET < 1
#error ARGS_PARAM_OFFSET must be at least 1 for proper formatting
#endif

#ifndef ARGS_HELP_OFFSET
/** @ingroup args_customizable
 * @brief Offset from longest argument for help text.
 * @remark Overridable before including @ref args.h.
 */
#define ARGS_HELP_OFFSET (4)
#elif ARGS_HELP_OFFSET < 1
#error ARGS_HELP_OFFSET must be at least 1 for proper formatting
#endif

#ifndef ARGS_IMPLICIT_SETS
/** @ingroup args_customizable
 * @brief Maximum implicit allocations of @ref argument::set booleans.
 * @remark Overridable before including @ref args.h.
 */
#define ARGS_IMPLICIT_SETS (64)
#elif ARGS_IMPLICIT_SETS < 1
#error ARGS_IMPLICIT_SETS must be at least 1 for defined behavior
#endif

/** @cond ARGS_INTERNALS */

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

/** @endcond */

#else

#endif /* ARGS_IMPLEMENTATION */

#endif /* UTIL_ARGS_H */
