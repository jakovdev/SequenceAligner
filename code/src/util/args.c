#include "util/args.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !(defined(args_perr) && defined(args_pdev) && defined(args_ierr) && \
      defined(args_abort))
#include "util/print.h"
#endif

#ifndef args_perr
#define args_perr(...) perror(__VA_ARGS__)
#endif

#ifndef args_pdev
#define args_pdev(...) pdev(__VA_ARGS__)
#endif

#ifndef args_ierr
#define args_ierr(arg) perror("Internal error for %s", arg_str(arg))
#endif

#ifndef args_abort
#define args_abort()            \
	do {                    \
		psection_end(); \
		abort();        \
	} while (0)
#endif

struct args_raw argr = { 0 };

/* Lists */
static struct argument *args;
static struct argument *help;
static struct argument *validate;
static struct argument *action;

#define for_each_arg(a, list) \
	for (struct argument *a = list; a; a = a->_.next_##list)
#define for_each_dep(a, dep)                                    \
	for (size_t dep##i = 0; dep##i < a->_.deps_n; dep##i++) \
		for (struct argument *dep = a->_.deps[dep##i]; dep; dep = NULL)
#define for_each_con(a, con)                                    \
	for (size_t con##i = 0; con##i < a->_.cons_n; con##i++) \
		for (struct argument *con = a->_.cons[con##i]; con; con = NULL)

static size_t args_num;
static size_t longest;

static const char *arg_str(const struct argument *a)
{
	if (!a)
		return "<null-arg>";

	/* Maximum of two arg_str calls in a row */
	static char buf[2][BUFSIZ];
	static size_t i = 0;

	if (a->opt && a->lopt)
		snprintf(buf[i], sizeof(buf[i]), "-%c, --%s", a->opt, a->lopt);
	else if (a->opt)
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
		args_pdev("ARGS_IMPLICIT_SETS exceeded, try increasing it");
		args_ierr(a);
		args_abort();
	}

	a->set = &sets[sets_n++];
}

void _args_register(struct argument *a)
{
	if (!a) {
		args_pdev("Cannot register %s", arg_str(a));
		args_ierr(a);
		args_abort();
	}

	if (!a->opt && !a->lopt) {
		args_pdev("%s must have an option", arg_str(a));
		args_ierr(a);
		args_abort();
	}

	if (a->_.valid) {
		args_pdev("%s has internals pre-set", arg_str(a));
		args_ierr(a);
		args_abort();
	}

	if (a->param_req != ARG_PARAM_NONE && !a->param) {
		args_pdev("%s requires parameter but .param=NULL", arg_str(a));
		args_ierr(a);
		args_abort();
	}

	if (a->param_req != ARG_PARAM_NONE && !a->parse_callback) {
		args_pdev("%s has .param but .parse_callback=NULL", arg_str(a));
		args_ierr(a);
		args_abort();
	}

	if (a->validate_phase != ARG_CALLBACK_ALWAYS && !a->validate_callback) {
		args_pdev("%s has .validate_phase but .validate_callback=NULL",
			  arg_str(a));
		args_ierr(a);
		args_abort();
	}

	if (a->action_phase != ARG_CALLBACK_ALWAYS && !a->action_callback) {
		args_pdev("%s has .action_phase but .action_callback=NULL",
			  arg_str(a));
		args_ierr(a);
		args_abort();
	}

	if (a->arg_req == ARG_SOMETIME && !a->_.deps && !a->_.cons &&
	    !a->validate_callback) {
		args_pdev("%s has no dependencies, conflicts, or validator",
			  arg_str(a));
		args_ierr(a);
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
		if (a->_.deps || a->_.cons)
			needs_set = true;
		if (needs_set)
			arg_set_new(a);
	}

	if (!a->_.deps) {
		if (a->_.deps_n > 0) {
			args_pdev("%s has deps_n=%zu but deps=NULL", arg_str(a),
				  a->_.deps_n);
			args_pdev("Add dependencies using ARG_DEPENDS()");
			args_ierr(a);
			args_abort();
		}

		if (a->_.deps_phase != ARG_RELATION_PARSE) {
			args_pdev("%s has relation phase but no dependencies",
				  arg_str(a));
			args_ierr(a);
			args_abort();
		}

		goto arg_no_deps;
	}

	size_t ndeps = 0;
	while (a->_.deps[ndeps])
		ndeps++;

	if (ndeps != a->_.deps_n) {
		args_pdev("%s deps_n=%zu but actual is %zu", arg_str(a),
			  a->_.deps_n, ndeps);
		args_pdev("Add dependencies using ARG_DEPENDS()");
		args_ierr(a);
		args_abort();
	}

	for_each_dep(a, dep) {
		if (!dep) {
			args_pdev("%s NULL deps[%zu]", arg_str(a), depi);
			args_ierr(a);
			args_abort();
		}

		if (dep == a) {
			args_pdev("%s depends on itself", arg_str(a));
			args_ierr(a);
			args_abort();
		}

		if (!dep->set)
			arg_set_new(dep);
	}

arg_no_deps:
	if (!a->_.cons) {
		if (a->_.cons_n > 0) {
			args_pdev("%s cons_n=%zu but cons=NULL", arg_str(a),
				  a->_.cons_n);
			args_pdev("Add conflicts using ARG_CONFLICTS()");
			args_ierr(a);
			args_abort();
		}

		if (a->_.cons_phase != ARG_RELATION_PARSE) {
			args_pdev("%s has relation phase but no conflicts",
				  arg_str(a));
			args_ierr(a);
			args_abort();
		}

		goto arg_no_cons;
	}

	size_t ncons = 0;
	while (a->_.cons[ncons])
		ncons++;

	if (ncons != a->_.cons_n) {
		args_pdev("%s cons_n=%zu but actual is %zu", arg_str(a),
			  a->_.cons_n, ncons);
		args_pdev("Add conflicts using ARG_CONFLICTS()");
		args_ierr(a);
		args_abort();
	}

	for_each_con(a, con) {
		if (!con) {
			args_pdev("%s NULL cons[%zu]", arg_str(a), coni);
			args_ierr(a);
			args_abort();
		}

		if (con == a) {
			args_pdev("%s conflicts itself", arg_str(a));
			args_ierr(a);
			args_abort();
		}

		if (!con->set)
			arg_set_new(con);

		for_each_dep(a, dep) {
			if (dep != con)
				continue;

			args_pdev("%s both depends and conflicts %s",
				  arg_str(a), arg_str(con));
			args_ierr(a);
			args_abort();
		}
	}

arg_no_cons:
	for_each_arg(c, args) {
		if ((a->opt && c->opt && a->opt == c->opt) ||
		    (a->lopt && c->lopt && strcmp(a->lopt, c->lopt) == 0)) {
			args_pdev("%s same opts as %s", arg_str(a), arg_str(c));
			args_ierr(a);
			args_abort();
		}
	}

	a->_.next_args = args;
	args = a;

#define args_insert(list)                                               \
	do {                                                            \
		if (!list || a->list##_weight >= list->list##_weight) { \
			a->_.next_##list = list;                        \
			list = a;                                       \
		} else {                                                \
			struct argument *cur = list;                    \
			while (cur->_.next_##list &&                    \
			       cur->_.next_##list->list##_weight >      \
				       a->list##_weight)                \
				cur = cur->_.next_##list;               \
			a->_.next_##list = cur->_.next_##list;          \
			cur->_.next_##list = a;                         \
		}                                                       \
	} while (0);

	args_insert(help);
	args_insert(validate);
	args_insert(action);
#undef args_insert

	size_t len = 2 + strlen(arg_str(a));
	if (a->param)
		len += 1 + strlen(a->param);
	a->_.help_len = len;
	if (len > longest)
		longest = len;

	a->_.valid = true;
	args_num++;
}

static bool arg_process(struct argument *a, const char *str)
{
	if (!a->_.valid) {
		args_pdev("%s has internals pre-set", arg_str(a));
		args_pdev("Please register arguments using ARGUMENT()");
		args_ierr(a);
		args_abort();
	}

	if (a->set && *a->set) {
		args_perr("Argument %s specified multiple times", arg_str(a));
		return false;
	}

	if (a->parse_callback) {
		struct arg_callback ret = a->parse_callback(str, a->dest);
		if (ret.error) {
			args_perr("%s: %s", arg_str(a), ret.error);
			return false;
		}
	}

	if (a->_.deps_phase == ARG_RELATION_PARSE) {
		for_each_dep(a, dep) {
			if (!*dep->set) {
				args_perr("%s requires %s to be set first",
					  arg_str(a), arg_str(dep));
				return false;
			}
		}
	}

	if (a->_.cons_phase == ARG_RELATION_PARSE) {
		for_each_con(a, con) {
			if (*con->set) {
				args_perr("%s conflicts with %s", arg_str(a),
					  arg_str(con));
				return false;
			}
		}
	}

	if (a->set)
		*a->set = true;

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
		args_perr("Unknown: --%.*s", (int)name_len, name);
		return false;
	}

	const char *str = NULL;
	if (a->param_req == ARG_PARAM_REQUIRED) {
		if (value) {
			str = value;
		} else if (*i + 1 < argr.c) {
			str = argr.v[++(*i)];
		} else {
			args_perr("--%s requires a parameter", a->lopt);
			return false;
		}
	} else if (a->param_req == ARG_PARAM_OPTIONAL) {
		if (value)
			str = value;
	} else {
		if (value) {
			args_perr("--%s does not take a parameter", a->lopt);
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
			args_perr("Unknown: -%c", opt);
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
				args_perr("-%c requires a parameter", opt);
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

bool args_parse(int argc, char *argv[])
{
	argr.c = argc;
	argr.v = argv;

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
			args_pdev("%s has internals pre-set", arg_str(a));
			args_pdev("Please register arguments using ARGUMENT()");
			args_ierr(a);
			args_abort();
		}

		if (a->arg_req == ARG_REQUIRED && !*a->set) {
			bool any_conflict_set = false;
			for_each_con(a, con) {
				if (*con->set) {
					any_conflict_set = true;
					break;
				}
			}
			if (!any_conflict_set) {
				args_perr("Missing required argument: %s",
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
			args_pdev("Unknown dependency relation phase in %s",
				  arg_str(a));
			args_ierr(a);
			break;
		}

		if (should_check_deps) {
			for_each_dep(a, dep) {
				if (*dep->set)
					continue;
				args_perr("%s requires %s to be set",
					  arg_str(a), arg_str(dep));
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
			args_pdev("Unknown conflict relation phase in %s",
				  arg_str(a));
			args_ierr(a);
			break;
		}

		if (should_check_cons) {
			for_each_con(a, con) {
				if (!*con->set)
					continue;
				args_perr("%s conflicts with %s", arg_str(a),
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
			args_pdev("Unknown .validate enum in %s", arg_str(a));
			args_ierr(a);
			args_abort();
		}

		if (should_validate) {
			struct arg_callback ret = a->validate_callback();
			if (ret.error) {
				args_perr("%s: %s", arg_str(a), ret.error);
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
			args_pdev("Unknown .action enum in %s", arg_str(a));
			args_ierr(a);
			args_abort();
		}

		if (should_act)
			a->action_callback();
	}
}

static void arg_print_help(const struct argument *a)
{
	printf("  %s", arg_str(a));
	if (a->param)
		printf(" %s", a->param);

	if (!a->help) {
		printf("\n");
		return;
	}

	size_t len = a->_.help_len;

	len = ((len / 8) + 1) * 8;
	size_t pad = longest > len ? longest - len : 0;

	const char *phelp = a->help;
	bool first = true;
	while (*phelp) {
		const char *nl = strchr(phelp, '\n');
		size_t line = nl ? (size_t)(nl - phelp) : strlen(phelp);

		if (first) {
			printf("%*s\t%.*s", (int)pad, "", (int)line, phelp);
			first = false;
		} else {
			printf("\n%*s%.*s", (int)longest, "", (int)line, phelp);
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
	longest = ((longest / 8) + 1) * 8;

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
	.opt = 'h',
	.lopt = "help",
	.help = "Display this help message",
	.parse_callback = print_help,
	.help_weight = UINT_MAX,
};
#endif
