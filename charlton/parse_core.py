# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# This file implements a simple "shunting yard algorithm" parser for infix
# languages with parentheses. It is used as the core of our parser for
# formulas, but is generic enough to be used for other purposes as well
# (e.g. parsing linear constraints). It just builds a parse tree; semantics
# are somebody else's problem.
# 
# Plus it spends energy on tracking where each item in the parse tree comes
# from, to allow high-quality error reporting.
#
# You are expected to provide an collection of Operators, and an iterator that
# provides Tokens. Each Operator should have a unique token_type (which is an
# arbitrary Python object), and each Token should have a matching token_type,
# or one of the special types Token.LPAREN, Token.RPAREN, or
# Token.ATOMIC_EXPR. Each Token is required to have a valid Origin attached,
# for error reporting.

__all__ = ["Token", "ParseNode", "Operator", "parse"]

from charlton import CharltonError
from charlton.origin import Origin

class _UniqueValue(object):
    def __init__(self, print_as):
        self._print_as = print_as

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._print_as)

class Token(object):
    LPAREN = _UniqueValue("LPAREN")
    RPAREN = _UniqueValue("RPAREN")
    ATOMIC_EXPR = _UniqueValue("ATOMIC_EXPR")

    def __init__(self, type, origin, extra=None):
        self.type = type
        self.origin = origin
        self.extra = extra

    def __repr__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__,
                                   self.type, self.origin, self.extra)

class ParseNode(object):
    def __init__(self, op, args, origin):
        self.op = op
        self.args = args
        self.origin = origin

    def __repr__(self):
        return "ParseNode(%r, %r)" % (self.op, self.args)

class Operator(object):
    def __init__(self, token_type, arity, precedence):
        self.token_type = token_type
        self.arity = arity
        self.precedence = precedence
        self.origin = None

    def with_origin(self, origin):
        new_op = self.__class__(self.token_type, self.arity, self.precedence)
        new_op.origin = origin
        return new_op

    def __repr__(self):
        return "%s(%r, %r, %r)" % (self.__class__.__name__,
                                   self.token_type, self.arity, self.precedence)

_open_paren = Operator(Token.LPAREN, -1, -9999999)

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

def _read_noun_context(token, c):
    if token.type == Token.LPAREN:
        c.op_stack.append(_open_paren.with_origin(token.origin))
        return True
    elif token.type in c.unary_ops:
        c.op_stack.append(c.unary_ops[token.type].with_origin(token.origin))
        return True
    elif token.type == Token.RPAREN or token.type in c.binary_ops:
        raise CharltonError("expected a noun, not '%s'"
                            % (token.origin.relevant_code(),),
                            token)
    elif token.type == Token.ATOMIC_EXPR:
        c.noun_stack.append(token)
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

def _read_op_context(token, c):
    if token.type == Token.RPAREN:
        while c.op_stack and c.op_stack[-1].token_type != Token.LPAREN:
            _run_op(c)
        if not c.op_stack:
            raise CharltonError("missing '(' or extra ')'", token)
        assert c.op_stack[-1].token_type == Token.LPAREN
        c.op_stack.pop()
        return False
    elif token.type in c.binary_ops:
        op = c.binary_ops[token.type].with_origin(token.origin)
        while (c.op_stack and op.precedence <= c.op_stack[-1].precedence):
            _run_op(c)
        c.op_stack.append(op)
        return True
    else:
        raise CharltonError("expected an operator", token)
    assert False

def parse(token_source, operators):
    token_source = iter(token_source)

    unary_ops = {}
    binary_ops = {}
    for op in operators:
        assert op.precedence > _open_paren.precedence
        if op.arity == 1:
            unary_ops[op.token_type] = op
        elif op.arity == 2:
            binary_ops[op.token_type] = op
        else:
            raise ValueError, "operators must be unary or binary"

    c = _ParseContext(unary_ops, binary_ops)

    # This is an implementation of Dijkstra's shunting yard algorithm:
    #   http://en.wikipedia.org/wiki/Shunting_yard_algorithm
    #   http://www.engr.mun.ca/~theo/Misc/exp_parsing.htm

    want_noun = True
    for token in token_source:
        if want_noun:
            want_noun = _read_noun_context(token, c)
        else:
            want_noun = _read_op_context(token, c)

    if want_noun:
        assert c.op_stack
        raise CharltonError("expected a noun, but instead the expression ended",
                            c.op_stack[-1])

    while c.op_stack:
        if c.op_stack[-1].token_type == Token.LPAREN:
            raise CharltonError("Unmatched '('", c.op_stack[-1])
        _run_op(c)

    assert len(c.noun_stack) == 1
    return c.noun_stack.pop()
