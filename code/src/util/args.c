#include "util/args.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/print.h"
#define pierr(arg) perr("Internal error for %s", arg_str(arg))

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
		pdev("ARGS_IMPLICIT_SETS exceeded, try increasing it");
		pierr(a);
		pabort();
	}

	a->set = &sets[sets_n++];
}

void _args_register(struct argument *a)
{
	if (!a) {
		pdev("Cannot register %s", arg_str(a));
		pierr(a);
		pabort();
	}

	if (!a->opt && !a->lopt) {
		pdev("%s must have an option", arg_str(a));
		pierr(a);
		pabort();
	}

	if (a->_.valid) {
		pdev("%s has internals pre-set", arg_str(a));
		pierr(a);
		pabort();
	}

	if (a->param_req != ARG_PARAM_NONE && !a->param) {
		pdev("%s requires parameter but .param=NULL", arg_str(a));
		pierr(a);
		pabort();
	}

	if (a->param_req != ARG_PARAM_NONE && !a->parse_callback) {
		pdev("%s has .param but .parse_callback=NULL", arg_str(a));
		pierr(a);
		pabort();
	}

	if (a->validate_phase != ARG_CALLBACK_ALWAYS && !a->validate_callback) {
		pdev("%s has .validate_phase but .validate_callback=NULL",
		     arg_str(a));
		pierr(a);
		pabort();
	}

	if (a->action_phase != ARG_CALLBACK_ALWAYS && !a->action_callback) {
		pdev("%s has .action_phase but .action_callback=NULL",
		     arg_str(a));
		pierr(a);
		pabort();
	}

	if (a->arg_req == ARG_SOMETIME && !a->_.deps && !a->_.cons &&
	    !a->validate_callback) {
		pdev("%s has no dependencies, conflicts, or validator",
		     arg_str(a));
		pierr(a);
		pabort();
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

	if (!a->_.deps) {
		if (a->_.deps_n > 0) {
			pdev("%s has deps_n=%zu but deps=NULL", arg_str(a),
			     a->_.deps_n);
			pdev("Add dependencies using ARG_DEPENDS()");
			pierr(a);
			pabort();
		}

		if (a->_.deps_phase != ARG_RELATION_PARSE) {
			pdev("%s has relation phase but no dependencies",
			     arg_str(a));
			pierr(a);
			pabort();
		}

		goto arg_no_deps;
	}

	size_t ndeps = 0;
	while (a->_.deps[ndeps])
		ndeps++;

	if (ndeps != a->_.deps_n) {
		pdev("%s deps_n=%zu but actual is %zu", arg_str(a), a->_.deps_n,
		     ndeps);
		pdev("Add dependencies using ARG_DEPENDS()");
		pierr(a);
		pabort();
	}

	for_each_rel(a, deps, dep) {
		if (!dep) {
			pdev("%s NULL deps[%zu]", arg_str(a), depi);
			pierr(a);
			pabort();
		}

		if (dep == a) {
			pdev("%s depends on itself", arg_str(a));
			pierr(a);
			pabort();
		}

		if (!dep->set)
			arg_set_new(dep);
	}

arg_no_deps:
	if (!a->_.cons) {
		if (a->_.cons_n > 0) {
			pdev("%s cons_n=%zu but cons=NULL", arg_str(a),
			     a->_.cons_n);
			pdev("Add conflicts using ARG_CONFLICTS()");
			pierr(a);
			pabort();
		}

		if (a->_.cons_phase != ARG_RELATION_PARSE) {
			pdev("%s has relation phase but no conflicts",
			     arg_str(a));
			pierr(a);
			pabort();
		}

		goto arg_no_cons;
	}

	size_t ncons = 0;
	while (a->_.cons[ncons])
		ncons++;

	if (ncons != a->_.cons_n) {
		pdev("%s cons_n=%zu but actual is %zu", arg_str(a), a->_.cons_n,
		     ncons);
		pdev("Add conflicts using ARG_CONFLICTS()");
		pierr(a);
		pabort();
	}

	for_each_rel(a, cons, con) {
		if (!con) {
			pdev("%s NULL cons[%zu]", arg_str(a), coni);
			pierr(a);
			pabort();
		}

		if (con == a) {
			pdev("%s conflicts itself", arg_str(a));
			pierr(a);
			pabort();
		}

		if (!con->set)
			arg_set_new(con);

		for_each_rel(a, deps, dep) {
			if (dep != con)
				continue;

			pdev("%s both depends and conflicts %s", arg_str(a),
			     arg_str(con));
			pierr(a);
			pabort();
		}
	}

arg_no_cons:
	if (!a->_.subs) {
		if (a->_.subs_n > 0) {
			pdev("%s subs_n=%zu but subs=NULL", arg_str(a),
			     a->_.subs_n);
			pdev("Specify subsets using ARG_SUBSETS()");
			pierr(a);
			pabort();
		}

		if (a->_.subs_strs) {
			pdev("%s has subs_strs but no subsets", arg_str(a));
			pierr(a);
			pabort();
		}

		goto arg_no_subs;
	}

	size_t nsubs = 0;
	while (a->_.subs[nsubs])
		nsubs++;

	if (nsubs != a->_.subs_n) {
		pdev("%s subs_n=%zu but actual is %zu", arg_str(a), a->_.subs_n,
		     nsubs);
		pdev("Specify subset args using ARG_SUBSETS()");
		pierr(a);
		pabort();
	}

	if (a->_.subs_strs) {
		size_t nsstrs = 0;
		while (a->_.subs_strs[nsstrs])
			nsstrs++;

		if (nsstrs != a->_.subs_n) {
			pdev("%s subs_n=%zu but subs_strs has %zu entries",
			     arg_str(a), a->_.subs_n, nsstrs);
			pdev("Both lists must be the same size");
			pierr(a);
			pabort();
		}
	}

	for_each_rel(a, subs, sub) {
		if (sub == a) {
			pdev("%s subsets itself", arg_str(a));
			pierr(a);
			pabort();
		}

		if (!sub->set)
			arg_set_new(sub);

		if (a->param_req != ARG_PARAM_REQUIRED &&
		    sub->param_req == ARG_PARAM_REQUIRED &&
		    (!a->_.subs_strs || a->_.subs_strs[subi] == ARG_SUBPASS)) {
			pdev("%s requires param but superset %s might not and has no custom string",
			     arg_str(sub), arg_str(a));
			pierr(a);
			pabort();
		}

		if (!a->set)
			arg_set_new(a);

		for_each_rel(a, cons, con) {
			if (con == sub) {
				pdev("%s both supersets and conflicts %s",
				     arg_str(a), arg_str(sub));
				pierr(a);
				pabort();
			}
		}

		for_each_rel(sub, deps, dep) {
			if (dep == a) {
				pdev("%s supersets %s but also depends on it",
				     arg_str(a), arg_str(sub));
				pierr(a);
				pabort();
			}
		}
	}

arg_no_subs:
	for_each_arg(c, args) {
		if ((a->opt && c->opt && a->opt == c->opt) ||
		    (a->lopt && c->lopt && strcmp(a->lopt, c->lopt) == 0)) {
			pdev("%s same opts as %s", arg_str(a), arg_str(c));
			pierr(a);
			pabort();
		}
	}

	a->_.next_args = args;
	args = a;

#define args_insert(list)                \
	do {                             \
		a->_.next_##list = list; \
		list = a;                \
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
		pdev("%s has internals pre-set", arg_str(a));
		pdev("Please register arguments using ARGUMENT()");
		pierr(a);
		pabort();
	}

	if (a->set && *a->set) {
		perr("Argument %s specified multiple times", arg_str(a));
		return false;
	}

	if (a->parse_callback) {
		struct arg_callback ret = a->parse_callback(str, a->dest);
		if (ret.error) {
			perr("%s: %s", arg_str(a), ret.error);
			return false;
		}
	}

	if (a->_.deps_phase == ARG_RELATION_PARSE) {
		for_each_rel(a, deps, dep) {
			if (!*dep->set) {
				perr("%s requires %s to be set first",
				     arg_str(a), arg_str(dep));
				return false;
			}
		}
	}

	if (a->_.cons_phase == ARG_RELATION_PARSE) {
		for_each_rel(a, cons, con) {
			if (*con->set) {
				perr("%s conflicts with %s", arg_str(a),
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
		perr("Unknown: --%.*s", (int)name_len, name);
		return false;
	}

	const char *str = NULL;
	if (a->param_req == ARG_PARAM_REQUIRED) {
		if (value) {
			str = value;
		} else if (*i + 1 < argr.c) {
			str = argr.v[++(*i)];
		} else {
			perr("--%s requires a parameter", a->lopt);
			return false;
		}
	} else if (a->param_req == ARG_PARAM_OPTIONAL) {
		if (value)
			str = value;
	} else {
		if (value) {
			perr("--%s does not take a parameter", a->lopt);
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
			perr("Unknown: -%c", opt);
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
				perr("-%c requires a parameter", opt);
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
				pdev("%s has invalid argument in " #list     \
				     "_order",                               \
				     arg_str(a));                            \
				pierr(a);                                    \
				pabort();                                    \
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

	args_reorder(help);
	args_reorder(validate);
	args_reorder(action);

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
			pdev("%s has internals pre-set", arg_str(a));
			pdev("Please register arguments using ARGUMENT()");
			pierr(a);
			pabort();
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
				perr("Missing required argument: %s",
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
			pdev("Unknown dependency relation phase in %s",
			     arg_str(a));
			pierr(a);
			break;
		}

		if (should_check_deps) {
			for_each_rel(a, deps, dep) {
				if (*dep->set)
					continue;
				perr("%s requires %s to be set", arg_str(a),
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
			pdev("Unknown conflict relation phase in %s",
			     arg_str(a));
			pierr(a);
			break;
		}

		if (should_check_cons) {
			for_each_rel(a, cons, con) {
				if (!*con->set)
					continue;
				perr("%s conflicts with %s", arg_str(a),
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
			pdev("Unknown .validate enum in %s", arg_str(a));
			pierr(a);
			pabort();
		}

		if (should_validate) {
			struct arg_callback ret = a->validate_callback();
			if (ret.error) {
				perr("%s: %s", arg_str(a), ret.error);
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
			pdev("Unknown .action enum in %s", arg_str(a));
			pierr(a);
			pabort();
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
	.help_order = ARG_ORDER_FIRST,
};
#endif
