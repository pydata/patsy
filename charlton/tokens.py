# This file is part of Charlton
# Copyright (C) 2011 Nathaniel Smith <njs@pobox.com>
# See file COPYING for license information.

# Utilities for dealing with Python code at the token level.
#
# Includes:
#   some core functionality for locating the origin of some object in the code
#     that it was parsed out of
#   a nice way to stream out tokens
#   a "pretty printer" that converts a sequence of tokens back into a
#       readable, white-space normalized string.
#   a utility function to calls to global functions with calls to other
#       functions

import tokenize
from cStringIO import StringIO

from charlton.origin import Origin, StringWithOrigin

__all__ = ["TokenSource", "pretty_untokenize", "replace_bare_funcall_tokens"]

class TokenSource(object):
    def __init__(self, formula_string):
        formula_file = StringIO(formula_string)
        self._formula_string = formula_string
        self._token_gen = tokenize.generate_tokens(formula_file.readline)
        self._next = []
        # We have our own end-of-stream handling, because TokenError handling
        # (below) means that we might cut off the stream before the tokenizer
        # itself raises StopIteration:
        self._done = False

    def __iter__(self):
        return self

    def peek(self):
        if not self._next:
            if self._done:
                raise ValueError, "can't consume tokens past end of stream"
            try:
                (token_type, token, (_, start), (_, end), code) = self._token_gen.next()
                token = StringWithOrigin(token, Origin(code, start, end))
                self._next.append((token_type, token))
            except StopIteration:
                raise ValueError, "stream ended without ENDMARKER?!?"
            except tokenize.TokenError, e:
                # TokenError is raised iff the tokenizer thinks that there is
                # some sort of multi-line construct in progress (e.g., an
                # unclosed parentheses, which in Python lets a virtual line
                # continue past the end of the physical line), and it hits the
                # end of the source text. We have our own error handling for
                # such cases, so just treat this as an end-of-stream.
                # 
                # Just in case someone adds some other error case:
                assert e.args[0].startswith("EOF in multi-line")
                self._next.append((tokenize.ENDMARKER, ""))
            if self._next[0][0] == tokenize.ENDMARKER:
                self._done = True
        return self._next[-1]

    def next(self):
        token = self.peek()
        self._next.pop()
        return token

    def push_back(self, token_type, token, origin=None):
        if not hasattr(token, "origin"):
            token = StringWithOrigin(token, origin)
        self._next.append((token_type, token))

def test_TokenSource():
    s = TokenSource("a + (b * -1)")
    for expected_token, start, end in [((tokenize.NAME, "a"), 0, 1),
                                       ((tokenize.OP, "+"), 2, 3),
                                       ((tokenize.OP, "("), 4, 5),
                                       ((tokenize.NAME, "b"), 5, 6),
                                       ((tokenize.OP, "*"), 7, 8),
                                       ((tokenize.OP, "-"), 9, 10),
                                       ((tokenize.NUMBER, "1"), 10, 11),
                                       ((tokenize.OP, ")"), 11, 12),
                                       ((tokenize.ENDMARKER, ""), None, None)]:
        assert s.peek() == expected_token
        assert s.peek() == expected_token
        if start is not None:
            _, token = s.peek()
            print repr(token.origin)
            assert token.origin.start == start
            assert token.origin.end == end
            assert token.origin.code == "a + (b * -1)"
        assert s.next() == expected_token
        if expected_token[0] != tokenize.ENDMARKER:
            s.push_back("foo", "bar", 1)
            assert s.peek() == ("foo", "bar")
            assert s.peek()[1].origin == 1
            s.next()
            s.peek()
            s.push_back("baz", "quux")
            assert s.next() == ("baz", "quux")
    from nose.tools import assert_raises
    assert_raises(ValueError, s.peek)
    assert_raises(ValueError, s.next)

_python_space_both = (list("+-*/%&^|<>")
                      + ["==", "<>", "!=", "<=", ">=",
                         "<<", ">>", "**", "//"])
_python_space_before = (_python_space_both
                        + ["!", "~"])
_python_space_after = (_python_space_both
                       + [",", ":"])

def pretty_untokenize(typed_tokens):
    text = []
    prev_was_space_delim = False
    prev_wants_space = False
    prev_was_open_paren_or_comma = False
    brackets = []
    for token_type, token in typed_tokens:
        assert token_type not in (tokenize.INDENT, tokenize.DEDENT,
                                  tokenize.NEWLINE, tokenize.NL)
        if token_type == tokenize.ENDMARKER:
            continue
        if token_type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
            if prev_wants_space or prev_was_space_delim:
                text.append(" ")
            text.append(token)
            prev_wants_space = False
            prev_was_space_delim = True
        else:
            if token in ("(", "[", "{"):
                brackets.append(token)
            elif brackets and token in (")", "]", "}"):
                brackets.pop()
            this_wants_space_before = (token in _python_space_before)
            this_wants_space_after = (token in _python_space_after)
            # Special case for slice syntax: foo[:10]
            # Otherwise ":" is spaced after, like: "{1: ...}", "if a: ..."
            if token == ":" and brackets and brackets[-1] == "[":
                this_wants_space_after = False
            # Special case for foo(*args), foo(a, *args):
            if token in ("*", "**") and prev_was_open_paren_or_comma:
                this_wants_space_before = False
                this_wants_space_after = False
            # Special case for "a = foo(b=1)":
            if token == "=" and not brackets:
                this_wants_space_before = True
                this_wants_space_after = True
            if prev_wants_space or this_wants_space_before:
                text.append(" ")
            text.append(token)
            prev_wants_space = this_wants_space_after
            prev_was_space_delim = False
        prev_was_open_paren_or_comma = token in ("(", ",")
    return "".join(text)

def normalize_token_spacing(code):
    tokens = [(t[0], t[1])
              for t in tokenize.generate_tokens(StringIO(code).readline)]
    return pretty_untokenize(tokens)

def test_pretty_untokenize_and_normalize_token_spacing():
    assert normalize_token_spacing("1 + 1") == "1 + 1"
    assert normalize_token_spacing("1+1") == "1 + 1"
    assert normalize_token_spacing("1*(2+3**2)") == "1 * (2 + 3 ** 2)"
    assert normalize_token_spacing("a and b") == "a and b"
    assert normalize_token_spacing("foo(a=bar.baz[1:])") == "foo(a=bar.baz[1:])"
    assert normalize_token_spacing("""{"hi":foo[:]}""") == """{"hi": foo[:]}"""
    assert normalize_token_spacing("""'a' "b" 'c'""") == """'a' "b" 'c'"""
    assert normalize_token_spacing('"""a""" is 1 or 2==3') == '"""a""" is 1 or 2 == 3'
    assert normalize_token_spacing("foo ( * args )") == "foo(*args)"
    assert normalize_token_spacing("foo ( a * args )") == "foo(a * args)"
    assert normalize_token_spacing("foo ( ** args )") == "foo(**args)"
    assert normalize_token_spacing("foo ( a ** args )") == "foo(a ** args)"
    assert normalize_token_spacing("foo (1, * args )") == "foo(1, *args)"
    assert normalize_token_spacing("foo (1, a * args )") == "foo(1, a * args)"
    assert normalize_token_spacing("foo (1, ** args )") == "foo(1, **args)"
    assert normalize_token_spacing("foo (1, a ** args )") == "foo(1, a ** args)"

    assert normalize_token_spacing("a=foo(b = 1)") == "a = foo(b=1)"

