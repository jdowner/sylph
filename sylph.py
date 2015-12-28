#!/usr/bin/env python3

import argparse
import logging
import math
import operator
import sys

log = logging.getLogger('sylph')


class UnknownVariable(Exception):
    pass


class Atom(object):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return repr("{}:{}".format(self.__class__.__name__.lower(), self.value))


class Expression(object):
    def __init__(self, terms):
        self.terms = terms

    def __iter__(self):
        return iter(self.terms)

    def __repr__(self):
        return "({})".format(' '.join(repr(t) for t in self.terms))

    @property
    def leader(self):
        return self.terms[0]

    @property
    def args(self):
        return self.terms[1:]


class Keyword(Atom):
    pass


class Symbol(Atom):
    pass


class Literal(Atom):
    pass


class Number(Literal):
    pass


class String(Literal):
    pass


class Procedure(object):
    def __call__(self, env, vals):
        pass


class InternalProcedure(object):
    def __call__(self, interp, env, *args):
        pass


class Define(InternalProcedure):
    def __call__(self, interp, env, args):
        assert len(args) == 2

        var = args[0]
        exp = args[1]

        if isinstance(var, Expression):
            var = interp.eval(var)

        env.set(var.value, interp.eval(exp))


class Lambda(InternalProcedure):
    def __call__(self, interp, env, args):
        assert len(args) == 2

        args, expr = args

        lambda_class = type('UserProc', (Procedure,), {})

        def impl(_, env, vals):
            local = dict(zip([a.value for a in args], vals))
            local = Env(local=local, outer=env)
            return interp.eval(expr, local)

        setattr(lambda_class, '__call__', impl)

        return lambda_class()


#keywords = (
#        'define',
#        'format',
#        'if',
#        'lambda',
#        'quote',
#        'set!',
#        )

def parse(program):
    "Read a Scheme expression from a string."
    for expression in read_from_token_iter(iter(tokenize(program))):
        yield expression


def tokenize(s):
    "Convert a string into a list of tokens."
    return s.replace('(',' ( ').replace(')',' ) ').split()


def read_from_token_iter(tokens):
    "Read an expression from a sequence of tokens."
    for token in tokens:
        if token == ')':
            break

        elif token == '(':
            yield Expression([t for t in read_from_token_iter(tokens)])

        else:
            try:
                yield Number(int(token))
                continue
            except ValueError:
                pass

            try:
                yield Number(float(token))
                continue
            except ValueError:
                pass

            yield Symbol(token)


def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env()

    def make_simple_proc(func):
        class SimpleProc(Procedure):
            def __call__(self, env, args):
                return func(*args)

        return SimpleProc()

    simple_procs = (
            ('+', make_simple_proc(operator.add)),
            ('-', make_simple_proc(operator.sub)),
            ('*', make_simple_proc(operator.mul)),
            ('/', make_simple_proc(operator.truediv)),
            ('>', make_simple_proc(operator.gt)),
            ('<', make_simple_proc(operator.lt)),
            ('>=', make_simple_proc(operator.ge)),
            ('<=', make_simple_proc(operator.le)),
            ('=', make_simple_proc(operator.eq)),
            ('eq?', make_simple_proc(operator.is_)),
            ('equal?', make_simple_proc(operator.eq)),
            ('not', make_simple_proc(operator.not_)),
            ('and', make_simple_proc(operator.and_)),
            ('or', make_simple_proc(operator.or_)),
            ('append', make_simple_proc(operator.add)),
            ('length', make_simple_proc(len)),
            ('abs', make_simple_proc(abs)),
            ('map', make_simple_proc(map)),
            ('max', make_simple_proc(max)),
            ('min', make_simple_proc(min)),
            ('round', make_simple_proc(round)),
            )

    for name, proc in simple_procs:
        env.set(name, proc)

    def proc_begin(interp, env, args):
        evaluations = [interp.eval(a, env) for a in args]
        return evaluations[-1]

    env.define_keyword('define', Define())
    env.define_keyword('lambda', Lambda())

#    env.update({
#        'apply': lambda _, __, x: [x[0](y) for y in x[1:]],
#        'begin': proc_begin,
#        'car': lambda _, __, x: x[0],
#        'cdr': lambda _, __, x: x[1:],
#        'cons': lambda _, __, x: [x[0]] + x[1:],
#        'list': lambda _, __, x: list(x),
#        'list?': lambda _, __, x: isinstance(x,list),
#        'null?': lambda _, __, x: x == [],
#        'number?': lambda _, __, x: isinstance(x, Number),
#        'procedure?': lambda _, __, x: callable(x),
#        'symbol?': lambda _, __, x: isinstance(x, Symbol),
#    })
    return env


class Env(object):
    "An environment: a dict of {'var':val} pairs, with an outer Env."
    def __init__(self, local=None, outer=None):
        self.vars = dict() if local is None else local
        self.keywords = set()
        self.outer = outer

    def __repr__(self):
        return repr(self.as_dict())

    def as_dict(self):
        vars = dict(self.vars)
        if self.outer is not None:
            vars.update(self.outer.as_dict())

        return vars

    def find(self, var):
        "Find the innermost Env where var appears."
        if var in self.vars:
            return self

        if self.outer is not None:
            return self.outer.find(var)

        raise UnknownVariable("unable to find '{}' variable".format(var))

    def get(self, key):
        if key in self.vars:
            return self.vars[key]

        if self.outer is not None:
            return self.outer.get(key)

        raise UnknownVariable("unable to get '{}' variable".format(key))

    def set(self, key, val):
        self.vars[key] = val

    def is_symbol(self, symbol):
        if symbol in self.vars:
            return True

        if self.outer is not None:
            return self.outer.is_symbol(symbol)

        return False

    def is_keyword(self, symbol):
        if symbol in self.keywords:
            return True

        if self.outer is not None:
            return self.outer.is_keyword(symbol)

        return False

    def define_keyword(self, keyword, proc):
        self.keywords.add(keyword)
        self.set(keyword, proc)


global_env = standard_env()


class InterpreterLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        indent = self.extra['indent']
        msg, kwargs = super().process(msg, kwargs)
        return '  ' * indent + msg, kwargs


class Interpreter(object):
    def __init__(self, verbose=False):
        self.indent = 0

    def eval(self, x, env=global_env):
        try:
            self.indent += 1
            log = InterpreterLoggerAdapter(
                    logging.getLogger('sylph'),
                    dict(indent=self.indent),
                    )

            if isinstance(x, Literal):
                log.debug("literal: {}".format(x))
                return x.value

            if isinstance(x, Symbol):
                log.debug("variable: {}".format(x))
                return env.get(x.value)

            if isinstance(x, Expression):
                log.debug("expr: {}".format(x))
                if env.is_symbol(x.leader.value):
                    # Retrieve the value of the symbol
                    val = env.get(x.leader.value)

                    if isinstance(val, InternalProcedure):
                        log.debug("iproc: {}".format(x))
                        return val(self, env, x.args)

                    elif isinstance(val, Procedure):
                        log.debug("eproc: {}".format(x))
                        return val(env, [self.eval(a, env) for a in x.args])

                    else:
                        return val

                else:
                    # The symbol is not a procedure so evaluate it and
                    # return the result. TODO what if there are trailing
                    # arguments?
                    return self.eval(x.leader, env)

            raise ValueError(x)

        finally:
            self.indent -= 1


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--repl', '-r', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('files', nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    # Make sure that the logger actually has a handler of some sort
    log.addHandler(logging.NullHandler())
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    env = standard_env()
    interpreter = Interpreter(verbose=args.verbose)

    for file in args.files:
        code = open(file).read()
        for expression in parse(code):
            print(interpreter.eval(expression, env))


if __name__ == "__main__":
    main()
