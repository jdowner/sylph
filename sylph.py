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


class Keyword(Atom):
    pass


class Number(Atom):
    pass


class Symbol(Atom):
    pass


class Procedure(object):
    def __call__(self, env, *args):
        pass


class Define(Procedure):
    def __call__(self, interp, env, args):
        var, exp = args
        env[var.value] = interp.eval(exp, env)


keywords = (
        'define',
        'format',
        'if',
        'lambda',
        'quote',
        'set!',
        )

def parse(program):
    "Read a Scheme expression from a string."
    for expression in read_from_token_iter(iter(tokenize(program))):
        yield expression

def tokenize(s):
    "Convert a string into a list of tokens."
    return s.replace('(',' ( ').replace(')',' ) ').split()


def atom(token):
    "Numbers become numbers; every other token is a symbol."
    try:
        return Number(int(token))
    except ValueError:
        pass

    try:
        return Number(float(token))
    except ValueError:
        pass

    return Keyword(token) if token in keywords else Symbol(token)

def read_from_token_iter(tokens):
    "Read an expression from a sequence of tokens."
    for token in tokens:
        if token == ')':
            break

        elif token == '(':
            yield [t for t in read_from_token_iter(tokens)]

        else:
            yield atom(token)


def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env()

    def make_simple_proc(func):
        class SimpleProc(Procedure):
            def __call__(self, interp, env, args):
                log.debug('make_simple_proc:args :{}'.format(args))
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
        env[name] = proc

    def proc_begin(interp, env, args):
        evaluations = [interp.eval(a, env) for a in args]
        return evaluations[-1]

    env.update({
        'apply': lambda _, __, x: [x[0](y) for y in x[1:]],
        'begin': proc_begin,
        'car': lambda _, __, x: x[0],
        'cdr': lambda _, __, x: x[1:],
        'cons': lambda _, __, x: [x[0]] + x[1:],
        'list': lambda _, __, x: list(x),
        'list?': lambda _, __, x: isinstance(x,list),
        'null?': lambda _, __, x: x == [],
        'number?': lambda _, __, x: isinstance(x, Number),
        'procedure?': lambda _, __, x: callable(x),
        'symbol?': lambda _, __, x: isinstance(x, Symbol),
    })
    return env


class Env(dict):
    "An environment: a dict of {'var':val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def find(self, var):
        "Find the innermost Env where var appears."
        if var in self:
            return self

        if self.outer is not None:
            return self.outer.find(var)

        raise UnknownVariable("unable to find '{}' variable".format(var))

    def get(self, key):
        "Find the innermost Env where var appears."
        if key in self:
            return self

        if self.outer is not None:
            return self.outer.find(key)

        raise UnknownVariable("unable to get '{}' variable".format(key))

    def set(self, key, val):
        if key in self:
            self[key] = val
            return

        if self.outer is not None:
            self.outer.set(key, val)
            return

        raise UnknownVariable("unable to set '{}' variable".format(key))


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

            "Evaluate an expression in an environment."
            if isinstance(x, Symbol):      # variable reference
                log.debug("variable: {}".format(x))
                return env.find(x.value)[x.value]

            if isinstance(x, Number):
                log.debug("number: {}".format(x.value))
                return x.value

            if not isinstance(x, list):  # constant literal
                log.debug("literal: {}".format(x))
                return x

            # If we get to here, we must be dealing with a list.
            key, args = x[0], x[1:]

            # If the list consists of one element, evaluate it
            if not args:
                return self.eval(key, env)

            if isinstance(key, Keyword):
                log.debug("keyword: {}".format(key))

                if key.value == 'quote':
                    log.debug("quote: {}".format(args))
                    return args

                if key.value == 'define':
                    var, exp = args
                    proc = Define()
                    proc(self, env, args)
                    return

                    log.debug("define: {} {}".format(var.value, exp))
                    env[var.value] = self.eval(exp, env)
                    return

                if key.value == 'set!':
                    var, exp = args
                    log.debug("set!: {} {}".format(var.value, exp))
                    env.set(var.value, self.eval(exp, env))
                    return

                if key.value == 'lambda':
                    parms, body = args
                    log.debug("lambda: {} {}".format(parms, body))

                    def procedure(*args):
                        return self.eval(body, Env([p.value for p in parms], args, env))

                    return procedure

                if key.value == 'if':
                    test, conseq, alt = args
                    exp = conseq if self.eval(test, env) else alt
                    log.debug("if: {} {} {}".format(test, conseq, alt))
                    return self.eval(exp, env)

                if key.value == 'format':
                    dest, text = args
                    text = self.eval(text, env)
                    log.debug("format: {} {}".format(dest, text))
                    print(text)
                    return

#            elif isinstance(key, Procedure):
#                log.debug('proc: {}'.format(key))
#                return key(self, env, args)

            else: # (proc arg...)
                log.debug('old proc: {}'.format(key))
                proc = self.eval(key, env)
                args = [self.eval(exp, env) for exp in args]

                if not callable(proc):
                    return [proc] + args

                return proc(self, env, args)

        finally:
            self.indent -= 1

_interpreter = Interpreter()
eval = _interpreter.eval


def parse_sexp(string):
    """
    >>> parse_sexp("(+ 5 (+ 3 5))")
    [['+', '5', ['+', '3', '5']]]

    """
    sexp = [[]]
    word = ''
    in_str = False
    for c in string:
        if c == '(' and not in_str:
            sexp.append([])
        elif c == ')' and not in_str:
            if word:
                sexp[-1].append(word)
                word = ''
            temp = sexp.pop()
            sexp[-1].append(temp)
        elif c in (' ', '\n', '\t') and not in_str:
            sexp[-1].append(word)
            word = ''
        elif c == '\"':
            in_str = not in_str
        else:
            word += c
    return sexp[0]

################ Interaction: A REPL

def repl(prompt='sylph> '):
    "A prompt-read-eval-print loop."
    interpreter = Interpreter()
    while True:
        try:
            val = interpreter.eval(parse(input(prompt)))
            if val is not None:
                print(lispstr(val))

        except (SystemExit, KeyboardInterrupt):
            break

def lispstr(exp):
    "Convert a Python object back into a Lisp-readable string."
    if  isinstance(exp, list):
        return '(' + ' '.join(map(lispstr, exp)) + ')'
    else:
        return str(exp)


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
            interpreter.eval(expression, env)



if __name__ == "__main__":
    main()
