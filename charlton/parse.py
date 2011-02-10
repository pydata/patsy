# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# This file defines a parser for a simple language based on S/R "formulas"
# (which are described in sections 2.3 and 2.4 in Chambers & Hastie, 1992).
#
# But basically, all this file cares about is parsing a simple language with:
#   -- infix operators of varying precedence
#   -- parentheses
#   -- leaf values are arbitrary python expressions
# It just builds a parse tree; semantics are someone else's problem.
# 
# Plus it spends energy on tracking where each item in the parse tree comes
# from, to allow high-quality error reporting.

import tokenize
from charlton import CharltonError
from charlton.origin import Origin, StringWithOrigin, CharltonErrorWithOrigin
from charlton.tokens import TokenSource, pretty_untokenize

__all__ = ["parse"]

class ParseNode(object):
    def __init__(self, op, args, origin):
        self.op = op
        self.args = args
        self.origin = origin

    def __repr__(self):
        return "ParseNode(%r, %r)" % (self.op, self.args)

class Operator(object):
    def __init__(self, token, arity, precedence):
        self.token = token
        self.arity = arity
        self.precedence = precedence
        self.origin = None

    def with_origin(self, origin):
        new_op = self.__class__(self.token, self.arity, self.precedence)
        new_op.origin = origin
        return new_op

    def __repr__(self):
        return "<Op %r>" % (self.token,)

_open_paren = Operator("(", -1, -9999999)

_default_ops = [
    Operator("~", 2, -100),
    Operator("~", 1, -100),

    Operator("+", 2, 100),
    Operator("-", 2, 100),
    Operator("*", 2, 200),
    Operator("/", 2, 200),
    Operator(":", 2, 300),
    Operator("**", 2, 500),

    Operator("+", 1, 100),
    Operator("-", 1, 100),
]

def _check_token(token_type, token):
    # These are filtered out of our input string, so they should never
    # appear...
    assert token_type not in (tokenize.NL, tokenize.NEWLINE)
    if token_type == tokenize.ERRORTOKEN:
        raise CharltonErrorWithOrigin("error tokenizing input "
                                      "(maybe an unclosed string?)",
                                      token)
    if token_type == tokenize.COMMENT:
        raise CharltonErrorWithOrigin("comments are not allowed", token)

class _ParseContext(object):
    def __init__(self, unary_ops, binary_ops):
        self.op_stack = []
        self.noun_stack = []
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops

def _combine_origin_attrs(objects):
    for obj in objects:
        assert obj.origin is not None
    return Origin.combine([obj.origin for obj in objects])

def _read_python_expr(token_source, c):
    end_tokens = set(c.binary_ops.keys()
                     + c.unary_ops.keys()
                     + [")"])
    token_types = []
    tokens = []
    bracket_level = 0
    while (bracket_level
           or (token_source.peek()[1] not in end_tokens
               and token_source.peek()[0] != tokenize.ENDMARKER)):
        assert bracket_level >= 0
        (token_type, token) = token_source.next()
        _check_token(token_type, token)
        if token in ("(", "[", "{"):
            bracket_level += 1
        if token in (")", "]", "}"):
            bracket_level -= 1
        if bracket_level < 0:
            raise CharltonErrorWithOrigin("unmatched close bracket",
                                          token)
        if token_type == tokenize.ENDMARKER:
            assert bracket_level > 0
            raise CharltonErrorWithOrigin("unclosed bracket in embedded "
                                          "Python expression",
                                          _combine_origin_attrs(tokens))
        token_types.append(token_type)
        tokens.append(token)
    text = pretty_untokenize(zip(token_types, tokens))
    return StringWithOrigin(text, _combine_origin_attrs(tokens))

def _read_noun_context(token_source, c):
    token_type, token = token_source.next()
    if token == "(":
        c.op_stack.append(_open_paren.with_origin(token.origin))
        return True
    elif token in c.unary_ops:
        c.op_stack.append(c.unary_ops[token].with_origin(token.origin))
        return True
    elif token == ")" or token in c.binary_ops:
        raise CharltonErrorWithOrigin("expected a noun, not '%s'" % (token,),
                                      token)
    elif token_type == tokenize.ENDMARKER:
        assert c.op_stack
        raise CharltonErrorWithOrigin("expected a noun, but the formula ended "
                                      "instead",
                                      c.op_stack[-1])
    elif token_type == tokenize.NUMBER:
        c.noun_stack.append(token)
        return False
    else:
        token_source.push_back(token_type, token)
        c.noun_stack.append(_read_python_expr(token_source, c))
        return False
    assert False

def _run_op(c):
    assert c.op_stack
    op = c.op_stack.pop()
    args = []
    for i in xrange(op.arity):
        args.append(c.noun_stack.pop())
    args.reverse()
    node = ParseNode(op, args, _combine_origin_attrs([op] + args))
    c.noun_stack.append(node)

def _read_op_context(token_source, c):
    token_type, token = token_source.next()
    assert token_type != tokenize.ENDMARKER
    if token == ")":
        while c.op_stack and c.op_stack[-1].token != "(":
            _run_op(c)
        if not c.op_stack:
            raise CharltonErrorWithOrigin("missing '(' or extra ')'",
                                          token)
        assert c.op_stack[-1].token == "("
        c.op_stack.pop()
        return False
    elif token in c.binary_ops:
        op = c.binary_ops[token].with_origin(token.origin)
        while (c.op_stack and op.precedence <= c.op_stack[-1].precedence):
            _run_op(c)
        c.op_stack.append(op)
        return True
    else:
        raise CharltonErrorWithOrigin("expected an operator", token)
    assert False

def parse(code, extra_operators=[]):
    code = code.replace("\n", " ").strip()
    if not code:
        code = "~ 1"
    token_source = TokenSource(code)

    for extra_operator in extra_operators:
        if extra_operator.precedence < 0:
            raise ValueError, "all operators must have precedence >= 0"

    all_op_list = _default_ops + extra_operators
    unary_ops = {}
    binary_ops = {}
    for op in all_op_list:
        if op.arity == 1:
            unary_ops[op.token] = op
        elif op.arity == 2:
            binary_ops[op.token] = op
        else:
            raise ValueError, "operators must be unary or binary"

    c = _ParseContext(unary_ops, binary_ops)

    # This is an implementation of Dijkstra's shunting yard algorithm:
    #   http://en.wikipedia.org/wiki/Shunting_yard_algorithm
    #   http://www.engr.mun.ca/~theo/Misc/exp_parsing.htm

    want_noun = True
    while True:
        if want_noun:
            want_noun = _read_noun_context(token_source, c)
        else:
            if token_source.peek()[0] == tokenize.ENDMARKER:
                break
            want_noun = _read_op_context(token_source, c)

    while c.op_stack:
        if c.op_stack[-1].token == "(":
            raise CharltonErrorWithOrigin("Unmatched '('",
                                          c.op_stack[-1])
        _run_op(c)

    assert len(c.noun_stack) == 1
    tree = c.noun_stack.pop()
    if not isinstance(tree, ParseNode) or tree.op.token != "~":
        tree = ParseNode(unary_ops["~"], [tree], tree.origin)
    return tree

#############

_parser_tests = {
    "": ["~", "1"],
    " ": ["~", "1"],
    " \n ": ["~", "1"],

    "1": ["~", "1"],
    "a": ["~", "a"],
    "a ~ b": ["~", "a", "b"],

    "(a ~ b)": ["~", "a", "b"],
    "a ~ ((((b))))": ["~", "a", "b"],
    "a ~ ((((+b))))": ["~", "a", ["+", "b"]],

    "a + b + c": ["~", ["+", ["+", "a", "b"], "c"]],
    "a + (b ~ c) + d": ["~", ["+", ["+", "a", ["~", "b", "c"]], "d"]],

    "a + np.log(a, base=10)": ["~", ["+", "a", "np.log(a, base=10)"]],
    # Note different spacing:
    "a + np . log(a , base = 10)": ["~", ["+", "a", "np.log(a, base=10)"]],

    # Check precedence
    "a + b ~ c * d": ["~", ["+", "a", "b"], ["*", "c", "d"]],
    "a + b * c": ["~", ["+", "a", ["*", "b", "c"]]],
    "-a**2": ["~", ["-", ["**", "a", "2"]]],
    "-a:b": ["~", ["-", [":", "a", "b"]]],
    "a + b:c": ["~", ["+", "a", [":", "b", "c"]]],
    "(a + b):c": ["~", [":", ["+", "a", "b"], "c"]],
    "a*b:c": ["~", ["*", "a", [":", "b", "c"]]],

    "a+b / c": ["~", ["+", "a", ["/", "b", "c"]]],
    "~ a": ["~", "a"],

    "-1": ["~", ["-", "1"]],
    }

def _compare_trees(got, expected):
    if isinstance(got, ParseNode):
        assert got.op.token == expected[0]
        for arg, expected_arg in zip(got.args, expected[1:]):
            _compare_trees(arg, expected_arg)
    else:
        assert got == expected

def _do_parse_test(test_cases, extra_operators):
    for code, expected in test_cases.iteritems():
        actual = parse(code, extra_operators=extra_operators)
        print repr(code), repr(expected)
        print actual
        _compare_trees(actual, expected)

def test_parse():
    _do_parse_test(_parser_tests, [])

def test_parse_origin():
    tree = parse("a ~ b + c")
    assert tree.origin == Origin("a ~ b + c", 0, 9)
    assert tree.op.origin == Origin("a ~ b + c", 2, 3)
    assert tree.args[0].origin == Origin("a ~ b + c", 0, 1)
    assert tree.args[1].origin == Origin("a ~ b + c", 4, 9)
    assert tree.args[1].op.origin == Origin("a ~ b + c", 6, 7)
    assert tree.args[1].args[0].origin == Origin("a ~ b + c", 4, 5)
    assert tree.args[1].args[1].origin == Origin("a ~ b + c", 8, 9)

# <> mark off where the error should be reported:
_parser_error_tests = [
    "a <+>",
    "a + <(>",

    "a + b <# asdf>",

    "<)>",
    "a + <)>",
    "<*> a",
    "a + <*>",

    "a + <foo[bar>",
    "a + <foo{bar>",
    "a + <foo(bar>",

    "a + <[bar>",
    "a + <{bar>",

    "a + <{bar[]>",

    "a + foo<]>bar",
    "a + foo[]<]>bar",
    "a + foo{}<}>bar",
    "a + foo<)>bar",

    "a + b<)>",
    "(a) <.>",

    "<(>a + b",

    "a +< >'foo", # Not the best placement for the error
]

# Split out so it can also be used by tests of the evaluator (which also
# raises CharltonErrorWithOrigin's)
def _parsing_error_test(parse_fn, error_descs):
    for error_desc in error_descs:
        letters = []
        start = None
        end = None
        for letter in error_desc:
            if letter == "<":
                start = len(letters)
            elif letter == ">":
                end = len(letters)
            else:
                letters.append(letter)
        bad_code = "".join(letters)
        assert start is not None and end is not None
        print error_desc
        print repr(bad_code), start, end
        try:
            parse_fn(bad_code)
        except CharltonErrorWithOrigin, e:
            print e
            assert e.origin.code == bad_code
            assert e.origin.start == start
            assert e.origin.end == end
        else:
            assert False, "parser failed to report an error!"

def test_parse_errors(extra_operators=[]):
    def parse_fn(code):
        return parse(code, extra_operators=extra_operators)
    _parsing_error_test(parse_fn, _parser_error_tests)

_extra_op_parser_tests = {
    "a | b": ["~", ["|", "a", "b"]],
    "a * b|c": ["~", ["*", "a", ["|", "b", "c"]]],
    }

def test_parse_extra_op():
    extra_operators = [Operator("|", 2, 250)]
    _do_parse_test(_parser_tests,
                   extra_operators=extra_operators)
    _do_parse_test(_extra_op_parser_tests,
                   extra_operators=extra_operators)
    test_parse_errors(extra_operators=extra_operators)
