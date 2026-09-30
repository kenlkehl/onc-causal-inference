"""Standalone OCI note-search interpreter, executed with Python -I -S.

Generated code runs only after a default-deny kernel syscall filter is active.
This file uses the standard library and the host's libseccomp shared library.
"""
import ast
import collections
import contextlib
import ctypes
import errno
import heapq
import io
import json
import math
import re
import resource
import sys


def restrict_process(memory_mb):
    """Allow in-memory computation and the inherited IPC pipes only."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    signatures = {
        "seccomp_init": (ctypes.c_void_p, [ctypes.c_uint32]),
        "seccomp_syscall_resolve_name": (ctypes.c_int, [ctypes.c_char_p]),
        "seccomp_rule_add": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]),
        "seccomp_load": (ctypes.c_int, [ctypes.c_void_p]),
        "seccomp_release": (None, [ctypes.c_void_p]),
    }
    for name, (restype, argtypes) in signatures.items():
        fn = getattr(seccomp, name)
        fn.restype, fn.argtypes = restype, argtypes
    allow, deny = 0x7FFF0000, 0x00050000 | errno.EPERM
    policy = seccomp.seccomp_init(deny)
    if not policy:
        raise RuntimeError("Cannot create syscall policy")
    # No filesystem opening, sockets, process creation, exec, ptrace, signals to
    # other processes, or policy-changing syscalls. File descriptors are pipes.
    permitted = (
        "read write close fstat lseek fcntl "
        "mmap mprotect munmap mremap madvise brk "
        "rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack "
        "clock_gettime gettimeofday time futex sched_yield getpid gettid "
        "restart_syscall exit exit_group"
    ).split()
    try:
        for syscall in permitted:
            number = seccomp.seccomp_syscall_resolve_name(syscall.encode("ascii"))
            if number >= 0 and seccomp.seccomp_rule_add(policy, allow, number, 0) != 0:
                raise RuntimeError("Cannot configure syscall policy")
        if seccomp.seccomp_load(policy) != 0:
            raise RuntimeError("Cannot activate syscall policy")
    finally:
        seccomp.seccomp_release(policy)


class BoundedText(io.TextIOBase):
    def __init__(self, limit):
        self.limit, self.text, self.truncated = limit, "", False

    def write(self, value):
        available = self.limit - len(self.text)
        self.text += value[:available]
        self.truncated |= len(value) > available
        return len(value)


class RecordTools:
    def __init__(self, history, max_output, max_patterns):
        self.history, self.max_output, self.max_patterns = history, max_output, max_patterns
        self.reset()

    def reset(self):
        self.spans = []
        self.omitted = False

    def remember(self, hit):
        span = [hit["start"], hit["end"]]
        if span[0] < span[1] and span not in self.spans:
            if len(self.spans) < 64:
                self.spans.append(span)
            else:
                self.omitted = True

    def read(self, start, end):
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.history):
            raise ValueError("read requires valid original character offsets")
        stop = min(end, start + max(1, self.max_output // 2))
        hit = {"start": start, "end": stop, "quote": self.history[start:stop], "truncated": stop < end}
        self.remember(hit)
        return hit

    def hit(self, a, b, context, allowance):
        width = min(max(1, allowance), len(self.history))
        left, right = max(0, a - context), min(len(self.history), b + context)
        if right - left > width:
            left = max(0, a - min(context, max(0, (width - (b - a)) // 2)))
            right = min(right, left + width)
        return {"start": left, "end": right, "quote": self.history[left:right],
                "match_start": a, "match_end": b,
                "truncated": left > max(0, a - context) or right < min(len(self.history), b + context)}

    def validate_bounds(self, context, limit, minimum):
        if type(context) is not int or not 0 <= context <= 2000 or type(limit) is not int or not minimum <= limit <= 20:
            raise ValueError("Invalid context or hit limit")

    def search(self, pattern, start=0, context=250, limit=6, flags=re.IGNORECASE):
        self.validate_bounds(context, limit, 1)
        if (not isinstance(pattern, str) or not 1 <= len(pattern) <= 2000
                or type(start) is not int or not 0 <= start <= len(self.history)):
            raise ValueError("Invalid regex or search start")
        found = re.compile(pattern, flags).finditer(self.history, start)
        hits, cursor = [], start
        for match in found:
            a, b = match.span()
            if a == b:
                raise ValueError("Search patterns must match nonempty text")
            if len(hits) == limit:
                return {"hits": hits, "has_more": True, "next_start": cursor}
            hit = self.hit(a, b, context, max(1, self.max_output // 2))
            hits.append(hit)
            self.remember(hit)
            cursor = b
        return {"hits": hits, "has_more": False, "next_start": None}

    def scan(self, patterns, *, context=160, limit=12):
        self.validate_bounds(context, limit, 2)
        if isinstance(patterns, str):
            patterns = [patterns]
        if (not isinstance(patterns, (list, tuple)) or not 1 <= len(patterns) <= self.max_patterns
                or any(not isinstance(p, str) or not 1 <= len(p) <= 2000 for p in patterns)):
            raise ValueError(f"scan accepts 1-{self.max_patterns} regex strings of at most 2000 characters")
        # Merge sorted match iterators; an overlap shared by several patterns is
        # counted once. A bounded reservoir covers positions across the record.
        streams = [re.compile(p, re.IGNORECASE).finditer(self.history) for p in patterns]
        occupied, sparse = {}, []
        previous, total = None, 0
        for match in heapq.merge(*streams, key=lambda m: m.span()):
            a, b = match.span()
            if a == b:
                raise ValueError("Scan patterns must match nonempty text")
            if (a, b) == previous:
                continue
            previous = (a, b)
            total += 1
            if total <= limit:
                sparse.append((a, b))
            cell = a * limit // max(1, len(self.history))
            if cell not in occupied:
                occupied[cell] = [(a, b), (a, b)]
            else:
                occupied[cell][1] = (a, b)
        choices = sparse if total <= limit else sorted({p for pair in occupied.values() for p in pair})
        if len(choices) > limit:
            choices = [choices[round(i * (len(choices) - 1) / (limit - 1))] for i in range(limit)]
        width = max(1, min(4000, (self.max_output - 300) // limit - 200))
        hits = [self.hit(a, b, context, width) for a, b in choices]
        result = {"hits": hits, "match_count": total, "omitted": total > len(hits),
                  "truncated": any(h["truncated"] for h in hits)}
        # Keep printed helper dictionaries intact at small output limits too.
        while hits and len(repr(result)) + 1 > self.max_output:
            hits.pop(len(hits) // 2)
            result.update(omitted=True, truncated=True)
        for hit in hits:
            self.remember(hit)
        return result


def serve():
    protocol_in, protocol_out = sys.stdin, sys.stdout
    try:
        setup = json.loads(protocol_in.readline())
        tools = RecordTools(setup["history"], setup["output_chars"], setup["patterns"])
        restrict_process(setup["memory_mb"])
    except Exception:
        protocol_out.write('{"ready":false}\n')
        protocol_out.flush()
        return
    state = {"history": setup["history"], "read": tools.read, "search": tools.search,
             "scan": tools.scan, "re": re, "json": json, "math": math, "collections": collections}
    protocol_out.write('{"ready":true}\n')
    protocol_out.flush()
    for line in protocol_in:
        tools.reset()
        capture = BoundedText(setup["output_chars"])
        error, detail = None, None
        try:
            code = json.loads(line)["code"]
            tree = ast.parse(code, mode="exec")
            trailing = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                exec(compile(tree, "<note-search>", "exec"), state)
                if trailing is not None:
                    value = eval(compile(ast.Expression(trailing.value), "<note-search>", "eval"), state)
                    if value is not None:
                        print(repr(value))
        except BaseException as exc:
            error, detail = type(exc).__name__[:100], str(exc)[:500]
        result = {"output": capture.text, "truncated": capture.truncated, "error": error,
                  "error_detail": detail, "source_spans": tools.spans, "sources_truncated": tools.omitted}
        protocol_out.write(json.dumps(result, ensure_ascii=True) + "\n")
        protocol_out.flush()


if __name__ == "__main__":
    serve()
