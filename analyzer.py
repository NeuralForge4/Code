import ast
import builtins
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from io import StringIO
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib

try:
    from radon.complexity import cc_rank, cc_visit
except ImportError:
    cc_visit = None
    cc_rank = None

try:
    import lizard
except ImportError:
    lizard = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "defect_classifier.joblib")
_LOADED_MODEL: Optional[Dict[str, Any]] = None

PYTHON_BUILTINS: Set[str] = set(dir(builtins)) | {
    "__name__", "__file__", "__doc__", "__package__", "__builtins__", "__debug__",
    "self", "cls", "args", "kwargs"
}

SHADOW_SENSITIVE_BUILTINS: Set[str] = {
    "list", "dict", "set", "tuple", "str", "int", "float", "bool", "id", "type",
    "input", "open", "len", "sum", "min", "max", "map", "filter", "zip", "range", "object"
}


def get_ml_artifact() -> Optional[Dict[str, Any]]:
    global _LOADED_MODEL
    if _LOADED_MODEL is None and os.path.exists(MODEL_PATH):
        try:
            _LOADED_MODEL = joblib.load(MODEL_PATH)
        except Exception:
            _LOADED_MODEL = None
    return _LOADED_MODEL


def detect_language(code: str, filename: str = "") -> str:
    ext = os.path.splitext(filename.strip())[1].lower() if filename else ""
    ext_map = {
        ".py": "python",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".java": "java",
    }
    if ext in ext_map:
        return ext_map[ext]

    if re.search(r"(\bpublic\s+class\b|\bprivate\s+class\b|\bSystem\.out\.print|\bimport\s+java\.|\bpublic\s+static\s+void\s+main\b|\bString\s+\w+)", code):
        return "java"
    if re.search(r"(#include\s*<(iostream|vector|string|cstring|memory|algorithm|cmath|fstream|sstream)>|\bstd::|\bcout\s*<<|\bcin\s*>>|\bnew\s+\w+\[|\bnamespace\s+\w+)", code):
        return "cpp"
    if re.search(r"(#include\s*<(stdio\.h|stdlib\.h|string\.h|math\.h|ctype\.h)>|\bprintf\s*\(|\bscanf\s*\(|\bmalloc\s*\(|\bfree\s*\(|\bgets\s*\()", code):
        return "c"
    if re.search(r"(^\s*def\s+\w+\s*\(|^\s*class\s+\w+.*:|^\s*import\s+\w+|^\s*from\s+\w+\s+import\b|\bprint\s*\(|\belif\b)", code, re.MULTILINE):
        return "python"
    return "python"


# =====================================================================
# MULTI-SYNTAX ERROR DETECTOR & DELIMITER CHECKER
# =====================================================================

def check_delimiter_balance(code: str, lang: str) -> List[Dict[str, Any]]:
    """Detects multiple unclosed or mismatched delimiters across the entire file."""
    findings: List[Dict[str, Any]] = []
    lines = code.splitlines()
    stack: List[Tuple[str, int, int]] = []  # (char, line, col)
    pairs = {")": "(", "]": "[", "}": "{"}

    in_line_comment = False
    in_block_comment = False
    in_str: Optional[str] = None
    escape = False

    for line_idx, line in enumerate(lines, start=1):
        col = 0
        while col < len(line):
            ch = line[col]

            # Comments
            if not in_str:
                if lang in {"c", "cpp", "java"}:
                    if not in_block_comment and line[col:col+2] == "//":
                        break
                    if not in_block_comment and line[col:col+2] == "/*":
                        in_block_comment = True
                        col += 2
                        continue
                    if in_block_comment and line[col:col+2] == "*/":
                        in_block_comment = False
                        col += 2
                        continue
                elif lang == "python":
                    if ch == "#":
                        break

            if in_block_comment:
                col += 1
                continue

            # Strings
            if not escape:
                if ch in {'"', "'"}:
                    if not in_str:
                        # Check triple quotes in python
                        if lang == "python" and line[col:col+3] in {'"""', "'''"}:
                            in_str = line[col:col+3]
                            col += 3
                            continue
                        in_str = ch
                    elif (len(in_str) == 3 and line[col:col+3] == in_str):
                        in_str = None
                        col += 3
                        continue
                    elif in_str == ch:
                        in_str = None
                elif ch == "\\" and in_str:
                    escape = True
            else:
                escape = False

            if not in_str:
                if ch in "([{":
                    stack.append((ch, line_idx, col + 1))
                elif ch in ")]}":
                    expected_open = pairs[ch]
                    if not stack:
                        findings.append({
                            "line": line_idx,
                            "severity": "critical",
                            "rule": f"{lang.upper()}-SYNTAX-DELIM",
                            "title": f"Unmatched closing delimiter '{ch}' with no opening '{expected_open}'",
                            "fix": f"Remove extra '{ch}' or add matching '{expected_open}'.",
                            "snippet": line.strip(),
                        })
                    else:
                        top_ch, top_line, _ = stack.pop()
                        if top_ch != expected_open:
                            findings.append({
                                "line": line_idx,
                                "severity": "critical",
                                "rule": f"{lang.upper()}-SYNTAX-DELIM",
                                "title": f"Mismatched delimiter: opened with '{top_ch}' on line {top_line} but closed with '{ch}'",
                                "fix": f"Change '{ch}' to match opening '{top_ch}'.",
                                "snippet": line.strip(),
                            })

            col += 1

    # Any remaining open delimiters
    for open_ch, open_line, _ in stack:
        close_ch = {v: k for k, v in pairs.items()}[open_ch]
        snippet = lines[open_line - 1].strip() if 1 <= open_line <= len(lines) else open_ch
        findings.append({
            "line": open_line,
            "severity": "critical",
            "rule": f"{lang.upper()}-SYNTAX-DELIM",
            "title": f"Unclosed delimiter '{open_ch}' (never closed with '{close_ch}')",
            "fix": f"Add closing '{close_ch}' to close block opened on line {open_line}.",
            "snippet": snippet,
        })

    return findings


# =====================================================================
# PYTHON ANALYZER
# =====================================================================

def _extract_target_names(target: ast.AST) -> Set[str]:
    names: Set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names |= _extract_target_names(elt)
    elif isinstance(target, ast.Starred):
        names |= _extract_target_names(target.value)
    return names


def _collect_defined_names(stmts: List[ast.AST]) -> Set[str]:
    defined: Set[str] = set()
    for node in stmts:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                defined |= _extract_target_names(t)
        elif isinstance(node, ast.AnnAssign):
            defined |= _extract_target_names(node.target)
        elif isinstance(node, ast.AugAssign):
            defined |= _extract_target_names(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            defined |= _extract_target_names(node.target)
            defined |= _collect_defined_names(node.body)
            defined |= _collect_defined_names(node.orelse)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars:
                    defined |= _extract_target_names(item.optional_vars)
            defined |= _collect_defined_names(node.body)
        elif isinstance(node, ast.If):
            defined |= _collect_defined_names(node.body)
            defined |= _collect_defined_names(node.orelse)
        elif isinstance(node, ast.While):
            defined |= _collect_defined_names(node.body)
            defined |= _collect_defined_names(node.orelse)
        elif isinstance(node, ast.Try):
            defined |= _collect_defined_names(node.body)
            for h in node.handlers:
                if h.name:
                    defined.add(h.name)
                defined |= _collect_defined_names(h.body)
            defined |= _collect_defined_names(node.orelse)
            defined |= _collect_defined_names(node.finalbody)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined.update(node.names)
    return defined


def _scan_python_syntax_and_indentation(code: str) -> List[Dict[str, Any]]:
    """Scans for multiple syntax and indentation errors before AST parsing."""
    findings: List[Dict[str, Any]] = []
    lines = code.splitlines()

    # Check missing colons on compound statements
    compound_pattern = re.compile(r"^\s*(def\s+\w+|class\s+\w+|if\b|elif\b|else|while\b|for\b|try|except\b|finally|with\b)[^:#\n]*$")
    for idx, line in enumerate(lines, start=1):
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        if compound_pattern.match(trimmed) and not trimmed.endswith(":"):
            findings.append({
                "line": idx,
                "severity": "critical",
                "rule": "PY-SYNTAX",
                "title": "SyntaxError: Missing colon ':' at end of statement header",
                "fix": f"Add ':' at the end of line {idx}.",
                "snippet": trimmed,
            })

    # Check tokenize errors (e.g. IndentationError, unclosed quotes)
    try:
        tokens = list(tokenize.generate_tokens(StringIO(code).readline))
    except tokenize.TokenError as te:
        msg = te.args[0]
        line_num = te.args[1][0] if len(te.args) > 1 and isinstance(te.args[1], tuple) else 1
        findings.append({
            "line": line_num,
            "severity": "critical",
            "rule": "PY-SYNTAX",
            "title": f"TokenError / SyntaxError: {msg}",
            "fix": "Fix unclosed string literal or quote delimiter.",
            "snippet": lines[line_num - 1].strip() if 1 <= line_num <= len(lines) else "",
        })
    except IndentationError as ie:
        ln = ie.lineno or 1
        findings.append({
            "line": ln,
            "severity": "critical",
            "rule": "PY-INDENT",
            "title": f"IndentationError: {ie.msg}",
            "fix": "Ensure consistent 4-space indentation across blocks.",
            "snippet": lines[ln - 1].strip() if 1 <= ln <= len(lines) else "",
        })
    except Exception:
        pass

    return findings


def _check_python_block_flow(stmts: List[ast.AST], lines: List[str], findings: List[Dict[str, Any]], depth: int = 1, line_offset: int = 0) -> None:
    if depth >= 4 and stmts:
        ln = getattr(stmts[0], "lineno", 1)
        findings.append({
            "line": ln,
            "severity": "warning",
            "rule": "PY-NEST01",
            "title": f"Deeply nested control block (nesting depth {depth})",
            "fix": "Use early returns (guard clauses) or extract helper functions to flatten logic.",
            "snippet": lines[ln - 1].strip() if 1 <= ln <= len(lines) else "nested block",
        })

    terminal_seen = False
    for stmt in stmts:
        if terminal_seen:
            ln = getattr(stmt, "lineno", 1)
            findings.append({
                "line": ln,
                "severity": "warning",
                "rule": "PY-DEAD01",
                "title": "Unreachable code detected after return / raise / break / continue",
                "fix": "Remove dead statements or move them before the terminating statement.",
                "snippet": lines[ln - 1].strip() if 1 <= ln <= len(lines) else "unreachable statement",
            })
            break

        if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            terminal_seen = True
        elif isinstance(stmt, ast.If):
            _check_python_block_flow(stmt.body, lines, findings, depth + 1)
            if stmt.orelse:
                _check_python_block_flow(stmt.orelse, lines, findings, depth + 1)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            _check_python_block_flow(stmt.body, lines, findings, depth + 1)
            if stmt.orelse:
                _check_python_block_flow(stmt.orelse, lines, findings, depth + 1)
        elif isinstance(stmt, ast.While):
            _check_python_block_flow(stmt.body, lines, findings, depth + 1)
            # Infinite while loop check
            is_always_true = (
                (isinstance(stmt.test, ast.Constant) and stmt.test.value is True)
                or (isinstance(stmt.test, ast.Constant) and stmt.test.value == 1)
            )
            if is_always_true:
                has_exit = any(
                    isinstance(sub, (ast.Break, ast.Return, ast.Raise))
                    or (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in {"exit", "quit"})
                    or (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in {"exit", "quit"})
                    for sub in ast.walk(stmt)
                )
                if not has_exit:
                    findings.append({
                        "line": stmt.lineno + line_offset,
                        "severity": "critical",
                        "rule": "PY-LOOP01",
                        "title": "Infinite 'while True:' loop with no break, return, or exit condition",
                        "fix": "Add a termination condition or break statement inside the loop body.",
                        "snippet": lines[stmt.lineno + line_offset - 1].strip() if (stmt.lineno + line_offset) <= len(lines) else "while True:",
                    })
        elif isinstance(stmt, ast.Try):
            _check_python_block_flow(stmt.body, lines, findings, depth + 1, line_offset=line_offset)
            for h in stmt.handlers:
                _check_python_block_flow(h.body, lines, findings, depth + 1, line_offset=line_offset)
            if stmt.finalbody:
                _check_python_block_flow(stmt.finalbody, lines, findings, depth + 1, line_offset=line_offset)


def _analyze_python_ast_node(tree: ast.AST, lines: List[str], findings: List[Dict[str, Any]], line_offset: int = 0) -> None:
    """Deeply inspects an AST tree (either full module or recovered function/block)."""
    # 1. Block control flow
    if hasattr(tree, "body") and isinstance(tree.body, list):
        _check_python_block_flow(tree.body, lines, findings, depth=1, line_offset=line_offset)

    imported_names: Dict[str, int] = {}
    has_star_import = any(
        isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
        for n in ast.walk(tree)
    )
    module_defined = _collect_defined_names(getattr(tree, "body", []))

    known_literal_lists: Dict[str, int] = {}
    known_dict_keys: Dict[str, Set[Any]] = {}
    zero_vars: Set[str] = set()
    used_names: Set[str] = set()
    assigned_vars: Dict[str, int] = {}

    # Check undefined variables in scope
    def check_undefined_in_scope(nodes: List[ast.AST], allowed_names: Set[str]) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    comp_vars: Set[str] = set()
                    for gen in sub.generators:
                        comp_vars |= _extract_target_names(gen.target)
                    allowed_names = allowed_names | comp_vars
                elif isinstance(sub, ast.Lambda):
                    lam_args = {a.arg for a in sub.args.args + sub.args.kwonlyargs}
                    if sub.args.vararg:
                        lam_args.add(sub.args.vararg.arg)
                    if sub.args.kwarg:
                        lam_args.add(sub.args.kwarg.arg)
                    allowed_names = allowed_names | lam_args
                elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if not has_star_import and sub.id not in allowed_names and sub.id not in PYTHON_BUILTINS:
                        ln = getattr(sub, "lineno", 1) + line_offset
                        findings.append({
                            "line": ln,
                            "severity": "critical",
                            "rule": "PY-UNDEF",
                            "title": f"Undefined variable or symbol '{sub.id}' (raises NameError)",
                            "fix": f"Define '{sub.id}', pass it as an argument, or import it before use.",
                            "snippet": lines[ln - 1].strip() if 1 <= ln <= len(lines) else sub.id,
                        })
                        allowed_names.add(sub.id)

    check_undefined_in_scope(getattr(tree, "body", []), set(module_defined))

    for node in ast.walk(tree):
        ln = getattr(node, "lineno", 1) + line_offset

        # Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                imported_names[name] = ln
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    name = alias.asname or alias.name
                    imported_names[name] = ln

        # Assignments
        elif isinstance(node, ast.Assign):
            # Zero assignment tracking
            if isinstance(node.value, ast.Constant) and node.value.value == 0:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        zero_vars.add(t.id)
            elif isinstance(node.value, ast.Constant) and node.value.value != 0:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        zero_vars.discard(t.id)

            # Literal list size tracking
            if isinstance(node.value, ast.List):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        known_literal_lists[t.id] = len(node.value.elts)

            # Literal dict keys tracking
            if isinstance(node.value, ast.Dict):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
                        known_dict_keys[t.id] = keys

            for t in node.targets:
                if isinstance(t, ast.Name):
                    if t.id in SHADOW_SENSITIVE_BUILTINS:
                        findings.append({
                            "line": ln,
                            "severity": "warning",
                            "rule": "PY-SHADOW",
                            "title": f"Variable '{t.id}' shadows built-in Python function/type '{t.id}()'",
                            "fix": f"Rename '{t.id}' (e.g. '{t.id}_val') so built-in {t.id}() remains usable.",
                            "snippet": lines[ln - 1].strip() if ln <= len(lines) else t.id,
                        })
                    if isinstance(node.value, ast.Name) and node.value.id == t.id:
                        findings.append({
                            "line": ln,
                            "severity": "warning",
                            "rule": "PY-SELF",
                            "title": f"Redundant self-assignment '{t.id} = {t.id}'",
                            "fix": "Remove redundant assignment or assign the intended value.",
                            "snippet": lines[ln - 1].strip() if ln <= len(lines) else f"{t.id} = {t.id}",
                        })
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and len(node.value.value) >= 6:
                        if re.search(r"(password|passwd|secret|api_key|apikey|token|auth_key)", t.id, re.IGNORECASE):
                            findings.append({
                                "line": ln,
                                "severity": "critical",
                                "rule": "PY-SEC02",
                                "title": f"Hardcoded credential/secret in variable '{t.id}'",
                                "fix": "Load sensitive credentials from os.environ or a secrets manager.",
                                "snippet": lines[ln - 1].strip() if ln <= len(lines) else t.id,
                            })

        # Name load/store
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            elif isinstance(node.ctx, ast.Store) and not node.id.startswith("_"):
                assigned_vars.setdefault(node.id, ln)

        # Division / Modulo by Zero
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                is_zero_lit = isinstance(node.right, ast.Constant) and node.right.value == 0
                is_zero_var = isinstance(node.right, ast.Name) and node.right.id in zero_vars
                if is_zero_lit or is_zero_var:
                    snippet = lines[ln - 1].strip() if ln <= len(lines) else "/ 0"
                    findings.append({
                        "line": ln,
                        "severity": "critical",
                        "rule": "PY-ZERO01",
                        "title": "Division or modulo by zero (raises ZeroDivisionError)",
                        "fix": "Validate that the divisor is non-zero before dividing.",
                        "snippet": snippet,
                    })

            # Type Error: Int + Str or other obvious type mismatches
            elif isinstance(node.op, ast.Add):
                left_is_num = isinstance(node.left, ast.Constant) and isinstance(node.left.value, (int, float))
                right_is_str = isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                left_is_str = isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
                right_is_num = isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
                if (left_is_num and right_is_str) or (left_is_str and right_is_num):
                    findings.append({
                        "line": ln,
                        "severity": "critical",
                        "rule": "PY-TYPE",
                        "title": "TypeError: Unsupported operand types for +: unsupported int and str concatenation",
                        "fix": "Convert integer to string via str() or use an f-string.",
                        "snippet": lines[ln - 1].strip() if ln <= len(lines) else "+",
                    })

        # Subscript bounds & dictionary keys
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and isinstance(node.slice, ast.Constant):
                var_name = node.value.id
                key_or_idx = node.slice.value
                # List index out of bounds
                if var_name in known_literal_lists and isinstance(key_or_idx, int):
                    size = known_literal_lists[var_name]
                    if key_or_idx >= size or key_or_idx < -size:
                        findings.append({
                            "line": ln,
                            "severity": "critical",
                            "rule": "PY-BOUNDS",
                            "title": f"IndexError: list index {key_or_idx} out of range for '{var_name}' (size {size})",
                            "fix": f"Use an index within bounds 0 to {size - 1}.",
                            "snippet": lines[ln - 1].strip() if ln <= len(lines) else f"{var_name}[{key_or_idx}]",
                        })
                # Dict key error
                elif var_name in known_dict_keys:
                    keys = known_dict_keys[var_name]
                    if keys and key_or_idx not in keys:
                        findings.append({
                            "line": ln,
                            "severity": "critical",
                            "rule": "PY-KEYERR",
                            "title": f"KeyError: Key {key_or_idx!r} not found in dictionary '{var_name}'",
                            "fix": f"Use {var_name}.get({key_or_idx!r}) with a default fallback.",
                            "snippet": lines[ln - 1].strip() if ln <= len(lines) else f"{var_name}[{key_or_idx}]",
                        })

        # Function Calls & Security
        elif isinstance(node, ast.Call):
            # Non-callable call (e.g. 123(), "string"())
            if isinstance(node.func, ast.Constant):
                findings.append({
                    "line": ln,
                    "severity": "critical",
                    "rule": "PY-TYPE",
                    "title": f"TypeError: '{type(node.func.value).__name__}' object is not callable",
                    "fix": "Check syntax; a constant value cannot be called as a function.",
                    "snippet": lines[ln - 1].strip() if ln <= len(lines) else "call",
                })
            elif isinstance(node.func, ast.Name):
                fn = node.func.id
                if fn in {"eval", "exec"}:
                    findings.append({
                        "line": ln,
                        "severity": "critical",
                        "rule": "PY-SEC01",
                        "title": f"Arbitrary code execution risk via {fn}()",
                        "fix": "Replace eval()/exec() with ast.literal_eval() or json.loads().",
                        "snippet": lines[ln - 1].strip() if ln <= len(lines) else fn,
                    })
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                obj_name = node.func.value.id if isinstance(node.func.value, ast.Name) else ""

                # SQL Injection checks
                if attr in {"execute", "executemany"}:
                    if node.args:
                        first_arg = node.args[0]
                        # check f-string, % format, or string concatenation
                        is_vuln_sql = False
                        if isinstance(first_arg, ast.JoinedStr):
                            is_vuln_sql = True
                        elif isinstance(first_arg, ast.BinOp) and isinstance(first_arg.op, (ast.Mod, ast.Add)):
                            is_vuln_sql = True
                        elif isinstance(first_arg, ast.Call) and isinstance(first_arg.func, ast.Attribute) and first_arg.func.attr == "format":
                            is_vuln_sql = True

                        if is_vuln_sql:
                            findings.append({
                                "line": ln,
                                "severity": "critical",
                                "rule": "PY-SQLI",
                                "title": "SQL Injection vulnerability: dynamic string formatted directly into query",
                                "fix": "Use parameterized queries (e.g. cursor.execute('SELECT ... WHERE id = ?', (val,))).",
                                "snippet": lines[ln - 1].strip() if ln <= len(lines) else f"execute(...)",
                            })

                if obj_name == "os" and attr == "system":
                    findings.append({
                        "line": ln,
                        "severity": "critical",
                        "rule": "PY-SEC03",
                        "title": "Command injection risk via os.system()",
                        "fix": "Use subprocess.run([...], shell=False) with an argument list.",
                        "snippet": lines[ln - 1].strip() if ln <= len(lines) else "os.system()",
                    })
                elif obj_name == "pickle" and attr in {"loads", "load"}:
                    findings.append({
                        "line": ln,
                        "severity": "critical",
                        "rule": "PY-SEC04",
                        "title": f"Unsafe deserialization via pickle.{attr}()",
                        "fix": "Use json.loads() for untrusted data instead of pickle.",
                        "snippet": lines[ln - 1].strip() if ln <= len(lines) else f"pickle.{attr}()",
                    })
                elif obj_name == "subprocess" and attr in {"run", "Popen", "call", "check_output", "check_call"}:
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            findings.append({
                                "line": ln,
                                "severity": "critical",
                                "rule": "PY-SEC05",
                                "title": f"subprocess.{attr}() invoked with shell=True allows shell injection",
                                "fix": "Pass command arguments as a list and set shell=False.",
                                "snippet": lines[ln - 1].strip() if ln <= len(lines) else f"subprocess.{attr}",
                            })

        # Exception Handling
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                findings.append({
                    "line": ln,
                    "severity": "warning",
                    "rule": "PY-ERR02",
                    "title": "Bare 'except:' catches SystemExit, KeyboardInterrupt, and MemoryError",
                    "fix": "Catch specific exception types (e.g. except (ValueError, KeyError) as exc:).",
                    "snippet": lines[ln - 1].strip() if ln <= len(lines) else "except:",
                })
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                findings.append({
                    "line": ln,
                    "severity": "warning",
                    "rule": "PY-ERR03",
                    "title": "Silent exception suppression ('except ...: pass') hides runtime failures",
                    "fix": "Log the error or handle recovery explicitly instead of passing silently.",
                    "snippet": lines[ln - 1].strip() if ln <= len(lines) else "except: pass",
                })

        # Function Definitions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_python_block_flow(node.body, lines, findings, depth=1, line_offset=line_offset)

            param_names = [a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
            if node.args.vararg:
                param_names.append(node.args.vararg.arg)
            if node.args.kwarg:
                param_names.append(node.args.kwarg.arg)

            fn_local_defined = _collect_defined_names(node.body) | set(param_names)
            check_undefined_in_scope(node.body, module_defined | fn_local_defined)

            # Check infinite recursion
            # Direct unconditional recursive call
            for sub_stmt in node.body:
                if isinstance(sub_stmt, ast.Return) and isinstance(sub_stmt.value, ast.Call):
                    if isinstance(sub_stmt.value.func, ast.Name) and sub_stmt.value.func.id == node.name:
                        # Unconditional recursion at top level of function
                        findings.append({
                            "line": sub_stmt.lineno + line_offset,
                            "severity": "critical",
                            "rule": "PY-REC01",
                            "title": f"Infinite recursion detected in '{node.name}()' (unconditional recursive call without base case)",
                            "fix": "Add a base-case return before making the recursive call.",
                            "snippet": lines[sub_stmt.lineno + line_offset - 1].strip() if (sub_stmt.lineno + line_offset) <= len(lines) else f"return {node.name}(...)",
                        })

            # Check inconsistent returns
            ret_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            has_value_ret = any(r.value is not None for r in ret_nodes)
            has_bare_ret = any(r.value is None for r in ret_nodes)
            ends_with_ret = bool(node.body) and isinstance(node.body[-1], (ast.Return, ast.Raise))
            if has_value_ret and (has_bare_ret or not ends_with_ret):
                findings.append({
                    "line": ln,
                    "severity": "warning",
                    "rule": "PY-RET01",
                    "title": f"Inconsistent return paths in '{node.name}()' (some branches return None implicitly)",
                    "fix": "Ensure all execution paths explicitly return a consistent type or raise an exception.",
                    "snippet": lines[ln - 1].strip() if ln <= len(lines) else f"def {node.name}",
                })

            # Mutable default args
            for default in node.args.defaults + node.args.kw_defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    findings.append({
                        "line": ln,
                        "severity": "warning",
                        "rule": "PY-MUT03",
                        "title": f"Mutable default argument in function '{node.name}'",
                        "fix": "Use None as the default value and initialize inside the function body.",
                        "snippet": lines[ln - 1].strip() if ln <= len(lines) else f"def {node.name}",
                    })
                    break

        # Comparison checks
        elif isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comp, ast.Constant) and comp.value in {True, False, None}:
                    findings.append({
                        "line": ln,
                        "severity": "style",
                        "rule": "PEP8-E712",
                        "title": f"Comparison to {comp.value} using == / !=",
                        "fix": "Use 'is None' / 'is not None' or direct boolean truthiness.",
                        "snippet": lines[ln - 1].strip() if ln <= len(lines) else "comparison",
                    })

    # Unused imports
    for mod, lineno in imported_names.items():
        if mod not in used_names:
            findings.append({
                "line": lineno,
                "severity": "style",
                "rule": "PY-F401",
                "title": f"Unused import '{mod}'",
                "fix": f"Remove unused import '{mod}' to keep dependencies clean.",
                "snippet": lines[lineno - 1].strip() if lineno <= len(lines) else f"import {mod}",
            })

    # Unused variables
    for var, lineno in assigned_vars.items():
        if var not in used_names and var not in imported_names:
            findings.append({
                "line": lineno,
                "severity": "style",
                "rule": "PY-F841",
                "title": f"Variable '{var}' is assigned but never used",
                "fix": f"Remove unused variable '{var}' or use it in computation.",
                "snippet": lines[lineno - 1].strip() if lineno <= len(lines) else var,
            })


def _analyze_python_with_recovery(code: str) -> List[Dict[str, Any]]:
    """Analyzes Python code, detecting multiple syntax errors and analyzing unaffected blocks."""
    findings: List[Dict[str, Any]] = []
    lines = code.splitlines()

    # 1. Delimiter balance check
    findings.extend(check_delimiter_balance(code, "python"))

    # 2. Syntax and indentation scanning
    findings.extend(_scan_python_syntax_and_indentation(code))

    # 3. Whole-file AST parse attempt
    parsed_full_ast = False
    try:
        tree = ast.parse(code)
        _analyze_python_ast_node(tree, lines, findings, line_offset=0)
        parsed_full_ast = True
    except SyntaxError as full_exc:
        # Full parse failed. Let's record the primary syntax error
        ln = full_exc.lineno or 1
        findings.append({
            "line": ln,
            "severity": "critical",
            "rule": "PY-SYNTAX",
            "title": f"Python SyntaxError: {full_exc.msg}",
            "fix": "Fix the syntax error on this line (check colons, brackets, quotes, or indentation).",
            "snippet": lines[ln - 1].strip() if 1 <= ln <= len(lines) else code[:60],
        })

    # 4. Block-Level AST Recovery: Split into top-level functions/classes and parse independently!
    if not parsed_full_ast:
        block_starts: List[int] = []
        for idx, line in enumerate(lines):
            if line.startswith(("def ", "class ", "@", "async def ")):
                block_starts.append(idx)

        if block_starts:
            ranges = []
            for i in range(len(block_starts)):
                start = block_starts[i]
                end = block_starts[i + 1] if i + 1 < len(block_starts) else len(lines)
                ranges.append((start, end))

            for start_idx, end_idx in ranges:
                block_lines = lines[start_idx:end_idx]
                block_code = "\n".join(block_lines)
                try:
                    block_tree = ast.parse(block_code)
                    # Successful recovery: analyze this block!
                    _analyze_python_ast_node(block_tree, lines, findings, line_offset=start_idx)
                except SyntaxError as b_exc:
                    b_ln = (b_exc.lineno or 1) + start_idx
                    if not any(f["line"] == b_ln and f["rule"] == "PY-SYNTAX" for f in findings):
                        findings.append({
                            "line": b_ln,
                            "severity": "critical",
                            "rule": "PY-SYNTAX",
                            "title": f"Python SyntaxError in block: {b_exc.msg}",
                            "fix": "Correct the syntax error in this function/class definition.",
                            "snippet": lines[b_ln - 1].strip() if 1 <= b_ln <= len(lines) else "",
                        })

    # 5. Resource leak check for unclosed files (e.g. open() without with)
    for idx, line in enumerate(lines, start=1):
        trimmed = line.strip()
        if re.search(r"\bopen\s*\(", trimmed) and not re.search(r"^\s*with\s+open\b", trimmed) and ".close()" not in code:
            findings.append({
                "line": idx,
                "severity": "warning",
                "rule": "PY-RES",
                "title": "Resource leak risk: file opened with open() without 'with' statement or close()",
                "fix": "Use 'with open(...) as f:' for automatic resource management.",
                "snippet": trimmed,
            })

    return findings


def _run_ruff(code: str) -> List[Dict[str, Any]]:
    """Runs Ruff if available via python -m ruff."""
    findings: List[Dict[str, Any]] = []
    lines = code.splitlines()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--stdin-filename", "submitted.py", "--output-format", "concise", "-"],
            input=code,
            text=True,
            capture_output=True,
            timeout=5,
        )
        for out_line in proc.stdout.splitlines():
            m = re.match(r"^submitted\.py:(\d+):\d+:\s+([A-Z0-9]+)\s+(.*)$", out_line.strip())
            if m:
                ln = int(m.group(1))
                rule_code = m.group(2)
                msg = m.group(3)
                findings.append({
                    "line": ln,
                    "severity": "style" if rule_code.startswith(("E", "W", "F401", "F841", "I")) else "warning",
                    "rule": f"RUFF-{rule_code}",
                    "title": msg,
                    "fix": "Apply Ruff recommended lint fix.",
                    "snippet": lines[ln - 1].strip() if 1 <= ln <= len(lines) else "",
                })
    except Exception:
        pass
    return findings


# =====================================================================
# C AND C++ DEEP STATIC & CONTROL-FLOW ANALYZER
# =====================================================================

def _analyze_c_cpp_deep(code: str, lang: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    lines = code.splitlines()

    # 1. Delimiter balance check
    findings.extend(check_delimiter_balance(code, lang))

    # Array bounds declarations: e.g. int arr[10]; or double v[5];
    fixed_arrays: Dict[str, Tuple[int, int]] = {}  # name -> (size, decl_line)
    for m in re.finditer(r"\b(?:int|long|short|char|float|double|bool|size_t)\s+([a-zA-Z_][\w]*)\s*\[\s*(\d+)\s*\]", code):
        fixed_arrays[m.group(1)] = (int(m.group(2)), 1)

    # String variables in C++
    cpp_strings: Dict[str, Tuple[int, int]] = {}  # name -> (length, line)
    if lang == "cpp":
        for idx, line in enumerate(lines, start=1):
            m = re.search(r'\bstd::string\s+([a-zA-Z_][\w]*)\s*=\s*"([^"]*)"', line)
            if m:
                cpp_strings[m.group(1)] = (len(m.group(2)), idx)

    # Pointer & Memory Lifecycle Tracking
    # Maps pointer_name -> (alloc_line, is_array, alloc_type)
    allocated_ptrs: Dict[str, Tuple[int, bool, str]] = {}
    freed_ptrs: Dict[str, int] = {}
    uninit_ptrs: Dict[str, int] = {}
    null_ptrs: Dict[str, int] = {}
    zero_vars: Set[str] = set()

    # Track functions and their return types
    fn_sig_re = re.compile(r"^\s*([\w:*&]+(?:\s+[\w:*&]+)*)\s+([a-zA-Z_][\w]*)\s*\(([^)]*)\)\s*(?:const)?\s*\{")
    current_fn_name: Optional[str] = None
    current_fn_ret_type: Optional[str] = None
    current_fn_has_return = False
    current_fn_start_line = 1
    brace_depth = 0

    fn_calls: List[Tuple[str, int]] = []

    in_block_comment = False
    for idx, line in enumerate(lines, start=1):
        trimmed = line.strip()
        if not trimmed:
            continue
        if "/*" in trimmed and "*/" not in trimmed:
            in_block_comment = True
            continue
        elif in_block_comment:
            if "*/" in trimmed:
                in_block_comment = False
            continue
        elif trimmed.startswith("/*") and trimmed.endswith("*/"):
            continue
        elif trimmed.startswith(("//", "#")):
            continue

        # Check function signature
        sig_m = fn_sig_re.match(line)
        if sig_m and brace_depth == 0:
            current_fn_ret_type = sig_m.group(1).strip()
            current_fn_name = sig_m.group(2).strip()
            current_fn_start_line = idx
            current_fn_has_return = False
            allocated_ptrs.clear()
            freed_ptrs.clear()

        brace_depth += trimmed.count("{") - trimmed.count("}")

        # Check function end
        if brace_depth == 0 and current_fn_name:
            # Check missing return statement in non-void function
            if current_fn_ret_type and current_fn_ret_type != "void" and current_fn_name != "main":
                if not current_fn_has_return:
                    findings.append({
                        "line": current_fn_start_line,
                        "severity": "critical",
                        "rule": f"{lang.upper()}-RET01",
                        "title": f"Missing return statement: non-void function '{current_fn_name}' does not return a value on all paths",
                        "fix": f"Add 'return <value>;' before exiting '{current_fn_name}'.",
                        "snippet": lines[current_fn_start_line - 1].strip(),
                    })

            # Check memory leak for any allocations not freed
            for ptr, (a_line, is_arr, a_type) in list(allocated_ptrs.items()):
                if ptr not in freed_ptrs:
                    rule = "CPP-LEAK" if lang == "cpp" else "CWE-401"
                    fix = f"Call 'delete[] {ptr};'" if is_arr else (f"Call 'delete {ptr};'" if a_type == "new" else f"Call 'free({ptr});'")
                    findings.append({
                        "line": a_line,
                        "severity": "critical",
                        "rule": rule,
                        "title": f"Memory leak: pointer '{ptr}' allocated on line {a_line} is never deallocated before function exits",
                        "fix": fix,
                        "snippet": lines[a_line - 1].strip() if a_line <= len(lines) else ptr,
                    })

            current_fn_name = None

        if "return" in trimmed:
            current_fn_has_return = True

            # Dangling reference / pointer return: return &stack_var;
            dangle_m = re.search(r"\breturn\s+&([a-zA-Z_][\w]*)\s*;", trimmed)
            if dangle_m:
                var_name = dangle_m.group(1)
                findings.append({
                    "line": idx,
                    "severity": "critical",
                    "rule": "CWE-562",
                    "title": f"Dangling pointer/reference: returning address of local stack variable '{var_name}'",
                    "fix": "Allocate on the heap (smart pointer/malloc) or return by value instead of returning stack address.",
                    "snippet": trimmed,
                })

        # Memory allocations in C: malloc, calloc, realloc
        if lang in {"c", "cpp"}:
            c_alloc_m = re.search(r"\b([a-zA-Z_][\w]*)\s*=\s*(?:\([^)]*\)\s*)?(malloc|calloc|realloc)\s*\(([^;]+)\)", trimmed)
            if c_alloc_m:
                p_name = c_alloc_m.group(1)
                alloc_fn = c_alloc_m.group(2)
                args = c_alloc_m.group(3)
                allocated_ptrs[p_name] = (idx, False, "malloc")
                freed_ptrs.pop(p_name, None)

                # sizeof(pointer) allocation bug check
                if re.search(rf"sizeof\s*\(\s*{re.escape(p_name)}\s*\)", args):
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "CWE-467",
                        "title": f"Incorrect allocation size: 'sizeof({p_name})' returns pointer size instead of object size",
                        "fix": f"Use 'sizeof(*{p_name})' or 'sizeof(<type>)' to allocate correct buffer size.",
                        "snippet": trimmed,
                    })

            # free(p) tracking
            free_m = re.search(r"\bfree\s*\(\s*([a-zA-Z_][\w]*)\s*\)", trimmed)
            if free_m:
                p_name = free_m.group(1)
                if p_name in freed_ptrs:
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "CWE-415",
                        "title": f"Double free() detected on pointer '{p_name}' (previously freed on line {freed_ptrs[p_name]})",
                        "fix": f"Set '{p_name} = NULL;' after freeing and avoid freeing twice.",
                        "snippet": trimmed,
                    })
                else:
                    freed_ptrs[p_name] = idx

                # In C++, check if free() was used on new/new[]
                if lang == "cpp" and p_name in allocated_ptrs and allocated_ptrs[p_name][2] == "new":
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "CPP-ALLOC-MISMATCH",
                        "title": f"Mismatched deallocation: pointer '{p_name}' was allocated with 'new' but deallocated with 'free()'",
                        "fix": f"Use 'delete' or 'delete[]' instead of 'free()'.",
                        "snippet": trimmed,
                    })

        # Memory allocations in C++: new, new[], delete, delete[]
        if lang == "cpp":
            # new[]
            new_arr_m = re.search(r"\b([a-zA-Z_][\w]*)\s*=\s*new\s+[\w:]+\s*\[", trimmed)
            if new_arr_m:
                p_name = new_arr_m.group(1)
                allocated_ptrs[p_name] = (idx, True, "new")
                freed_ptrs.pop(p_name, None)

            # scalar new
            new_scalar_m = re.search(r"\b([a-zA-Z_][\w]*)\s*=\s*new\s+[\w:]+\s*\(", trimmed)
            if new_scalar_m:
                p_name = new_scalar_m.group(1)
                allocated_ptrs[p_name] = (idx, False, "new")
                freed_ptrs.pop(p_name, None)

            # delete[]
            delete_arr_m = re.search(r"\bdelete\s*\[\]\s*([a-zA-Z_][\w]*)", trimmed)
            if delete_arr_m:
                p_name = delete_arr_m.group(1)
                if p_name in allocated_ptrs and not allocated_ptrs[p_name][1]:
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "CPP-DEL-MISMATCH",
                        "title": f"Mismatched deallocation: scalar object '{p_name}' deleted with 'delete[]'",
                        "fix": f"Use scalar 'delete {p_name};' instead of 'delete[]'.",
                        "snippet": trimmed,
                    })
                freed_ptrs[p_name] = idx

            # scalar delete
            delete_scalar_m = re.search(r"\bdelete\s+(?!\[\])([a-zA-Z_][\w]*)", trimmed)
            if delete_scalar_m:
                p_name = delete_scalar_m.group(1)
                if p_name in allocated_ptrs and allocated_ptrs[p_name][1]:
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "CPP-DEL-MISMATCH",
                        "title": f"Mismatched deallocation: array allocated with 'new[]' deleted with scalar 'delete'",
                        "fix": f"Use array 'delete[] {p_name};' instead of scalar 'delete'.",
                        "snippet": trimmed,
                    })
                freed_ptrs[p_name] = idx

        # Use-After-Free / Use-After-Delete Check
        for p_name, f_line in list(freed_ptrs.items()):
            if f_line < idx:
                # check dereference or indexing: *p, p->field, p[index]
                if re.search(rf"(?:\*\s*{re.escape(p_name)}\b|\b{re.escape(p_name)}\s*(?:->|\[))", trimmed):
                    rule = "CWE-416"
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": rule,
                        "title": f"Use-after-free / Use-after-delete on pointer '{p_name}' (freed on line {f_line})",
                        "fix": f"Do not dereference or access '{p_name}' after freeing/deleting it.",
                        "snippet": trimmed,
                    })
                    freed_ptrs.pop(p_name, None)

        # Uninitialized pointers: int *p; (without assignment)
        uninit_p_m = re.match(r"^(?:int|char|float|double|void|size_t)\s*\*\s*([a-zA-Z_][\w]*)\s*;", trimmed)
        if uninit_p_m:
            uninit_ptrs[uninit_p_m.group(1)] = idx

        for p_name, u_line in list(uninit_ptrs.items()):
            if u_line < idx:
                if re.search(rf"\b{re.escape(p_name)}\s*=(?!=)", trimmed):
                    uninit_ptrs.pop(p_name, None)
                elif re.search(rf"(?:\*\s*{re.escape(p_name)}\b|\b{re.escape(p_name)}\s*->)", trimmed):
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "CWE-824",
                        "title": f"Uninitialized pointer dereference: '{p_name}' is dereferenced before assignment",
                        "fix": f"Initialize '{p_name}' to NULL or allocate valid memory before dereferencing.",
                        "snippet": trimmed,
                    })
                    uninit_ptrs.pop(p_name, None)

        # NULL / nullptr dereference
        null_m = re.search(r"\b([a-zA-Z_][\w]*)\s*=\s*(?:NULL|nullptr|0)\s*;", trimmed)
        if null_m:
            null_ptrs[null_m.group(1)] = idx
        for ptr, n_line in list(null_ptrs.items()):
            if n_line < idx:
                if re.search(rf"\b{re.escape(ptr)}\s*=\s*(?!NULL|nullptr|0\b)", trimmed):
                    null_ptrs.pop(ptr, None)
                elif re.search(rf"(?:\*\s*{re.escape(ptr)}\b|\b{re.escape(ptr)}\s*(?:->|\[))", trimmed):
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "CWE-476",
                        "title": f"Potential NULL pointer dereference on '{ptr}' (set to NULL on line {n_line})",
                        "fix": f"Check 'if ({ptr} != NULL)' before dereferencing.",
                        "snippet": trimmed,
                    })
                    null_ptrs.pop(ptr, None)

        # Static Array Out-of-Bounds
        for arr_m in re.finditer(r"\b([a-zA-Z_][\w]*)\s*\[\s*(\d+)\s*\]", trimmed):
            arr_name = arr_m.group(1)
            idx_val = int(arr_m.group(2))
            is_decl = re.search(rf"\b(?:int|long|short|char|float|double|bool|size_t)\s+{re.escape(arr_name)}\s*\[", trimmed)
            if arr_name in fixed_arrays and not is_decl:
                size = fixed_arrays[arr_name][0]
                if idx_val >= size:
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "CWE-787",
                        "title": f"Array index {idx_val} out of bounds for '{arr_name}[{size}]' (valid indices: 0..{size - 1})",
                        "fix": f"Ensure index is strictly less than array capacity {size}.",
                        "snippet": trimmed,
                    })

        # Loop bound check: for (int i = 0; i <= size; i++) arr[i]
        loop_bound_m = re.search(r"for\s*\([^;]*;\s*([a-zA-Z_][\w]*)\s*<=\s*(\d+)\s*;", trimmed)
        if loop_bound_m:
            bound_val = int(loop_bound_m.group(2))
            for arr_name, (size, _) in fixed_arrays.items():
                if bound_val >= size:
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "CWE-193",
                        "title": f"Off-by-one array access: loop condition '<= {bound_val}' exceeds bounds of '{arr_name}[{size}]'",
                        "fix": f"Change '<= {bound_val}' to '< {size}'.",
                        "snippet": trimmed,
                    })

        # C++ Invalid String Indexing
        if lang == "cpp":
            for str_m in re.finditer(r"\b([a-zA-Z_][\w]*)\s*\[\s*(\d+)\s*\]", trimmed):
                s_name = str_m.group(1)
                s_idx = int(str_m.group(2))
                if s_name in cpp_strings:
                    s_len = cpp_strings[s_name][0]
                    if s_idx >= s_len:
                        findings.append({
                            "line": idx,
                            "severity": "critical",
                            "rule": "CPP-STR-BOUNDS",
                            "title": f"String index {s_idx} out of range for std::string '{s_name}' (length {s_len})",
                            "fix": f"Use an index within bounds 0 to {s_len - 1}.",
                            "snippet": trimmed,
                        })

        # Infinite loops
        # while (1) / while (true) / for (;;)
        if re.search(r"\b(?:while\s*\(\s*(?:1|true)\s*\)|for\s*\(\s*;\s*;\s*\))", trimmed):
            # check if body has break, return, or exit
            loop_body = []
            loop_depth = 0
            started = False
            for forward_line in lines[idx - 1:]:
                loop_depth += forward_line.count("{") - forward_line.count("}")
                if "{" in forward_line:
                    started = True
                loop_body.append(forward_line)
                if started and loop_depth <= 0:
                    break
            loop_text = "\n".join(loop_body)
            has_exit = re.search(r"\b(break|return|exit|goto)\b", loop_text)
            if not has_exit:
                findings.append({
                    "line": idx,
                    "severity": "critical",
                    "rule": "C-LOOP-INF",
                    "title": "Infinite loop detected with no break, return, or exit termination",
                    "fix": "Add a loop termination condition or break statement inside the body.",
                    "snippet": trimmed,
                })

        # for loop counter moving wrong way
        for_loop_m = re.search(r"\bfor\s*\([^;]*;\s*([a-zA-Z_][\w]*)\s*(<=|<|>=|>)\s*[^;]+;\s*\1(\+\+|--)", trimmed)
        if for_loop_m:
            op, step = for_loop_m.group(2), for_loop_m.group(3)
            if (op in {"<=", "<"} and step == "--") or (op in {">=", ">"} and step == "++"):
                findings.append({
                    "line": idx,
                    "severity": "critical",
                    "rule": "LOGIC-LOOP",
                    "title": f"Infinite loop condition: loop variable moves away from boundary condition ('{op}' with '{step}')",
                    "fix": "Change '--' to '++' (or reverse the loop boundary condition).",
                    "snippet": trimmed,
                })

        # Infinite recursion
        if current_fn_name:
            rec_call_m = re.search(rf"\b{re.escape(current_fn_name)}\s*\(", trimmed)
            if rec_call_m and "return" in trimmed and not re.search(r"\b(if|else|switch)\b", trimmed):
                if not current_fn_has_return or idx == current_fn_start_line + 1:
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "C-REC-INF",
                        "title": f"Infinite recursion detected in '{current_fn_name}()' (unconditional recursive call)",
                        "fix": "Add a base-case check with return before invoking the recursive function.",
                        "snippet": trimmed,
                    })

        # Division by zero
        if re.search(r"[/]\s*0(?![.\d])", trimmed) or re.search(r"[%]\s*0(?![.\d])", trimmed):
            findings.append({
                "line": idx,
                "severity": "critical",
                "rule": "CWE-369",
                "title": "Division or modulo by zero literal",
                "fix": "Ensure the divisor is non-zero before dividing.",
                "snippet": trimmed,
            })

        # Banned & Vulnerable C Functions
        if lang in {"c", "cpp"}:
            if re.search(r"\bgets\s*\(", trimmed):
                findings.append({
                    "line": idx,
                    "severity": "critical",
                    "rule": "CWE-242",
                    "title": "Banned function gets() causes buffer overflow vulnerability",
                    "fix": "Replace gets(buf) with bounded fgets(buf, sizeof(buf), stdin).",
                    "snippet": trimmed,
                })
            if re.search(r"\b(strcpy|strcat|sprintf)\s*\(", trimmed):
                findings.append({
                    "line": idx,
                    "severity": "critical",
                    "rule": "CWE-120",
                    "title": "Unbounded string copy/format operation causes buffer overflow vulnerability",
                    "fix": "Use strncpy(), strncat(), snprintf(), or C++ std::string.",
                    "snippet": trimmed,
                })
            if re.search(r'\bscanf\s*\(\s*"%s"', trimmed):
                findings.append({
                    "line": idx,
                    "severity": "critical",
                    "rule": "CWE-120",
                    "title": "Unbounded scanf(\"%s\") can overflow destination buffer",
                    "fix": "Specify field width (e.g. scanf(\"%63s\", buf)) or use fgets().",
                    "snippet": trimmed,
                })
            if re.search(r"\bprintf\s*\(\s*[a-zA-Z_][\w]*\s*\)\s*;", trimmed):
                findings.append({
                    "line": idx,
                    "severity": "critical",
                    "rule": "CWE-134",
                    "title": "Format string vulnerability: variable passed directly as printf format argument",
                    "fix": "Use printf(\"%s\", var) instead of printf(var).",
                    "snippet": trimmed,
                })

        # C++ File Handling: unchecked stream open
        if lang == "cpp":
            stream_m = re.search(r"\bstd::(?:ifstream|ofstream|fstream)\s+([a-zA-Z_][\w]*)\s*\(([^)]+)\)", trimmed)
            if stream_m:
                s_var = stream_m.group(1)
                # check if .is_open() or !s_var is tested in active code (excluding comments)
                code_no_comments = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
                if not re.search(rf"\b{re.escape(s_var)}\.is_open\(\)", code_no_comments) and not re.search(rf"!\s*{re.escape(s_var)}\b", code_no_comments):
                    findings.append({
                        "line": idx,
                        "severity": "warning",
                        "rule": "CPP-STREAM-CHECK",
                        "title": f"Unchecked file stream: '{s_var}' opened without verifying is_open()",
                        "fix": f"Check 'if (!{s_var}.is_open())' before performing stream I/O.",
                        "snippet": trimmed,
                    })

    # Missing semicolons check
    in_block_comment = False
    for idx, line in enumerate(lines, start=1):
        trimmed = line.strip()
        if not trimmed:
            continue
        if "/*" in trimmed and "*/" not in trimmed:
            in_block_comment = True
            continue
        elif in_block_comment:
            if "*/" in trimmed:
                in_block_comment = False
            continue
        elif trimmed.startswith("/*") and trimmed.endswith("*/"):
            continue
        elif trimmed.startswith(("//", "#")):
            continue
        line_clean = re.sub(r"//.*$", "", trimmed).strip()
        needs_semi = (
            not line_clean.endswith((";", "{", "}", ":", ",", "\\"))
            and not re.match(r"^(if|else|for|while|switch|try|catch|public|private|protected|class|struct|namespace|template|package|@)\b", line_clean)
            and not fn_sig_re.match(line_clean)
        )
        if needs_semi:
            findings.append({
                "line": idx,
                "severity": "critical",
                "rule": f"{lang.upper()}-SEMI",
                "title": "Missing semicolon ';' at end of statement",
                "fix": f"Add ';' at the end of line {idx}.",
                "snippet": trimmed,
            })

    return findings


# =====================================================================
# JAVA DEEP STATIC & CONTROL-FLOW ANALYZER
# =====================================================================

def _analyze_java_deep(code: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    lines = code.splitlines()

    # 1. Delimiter balance check
    findings.extend(check_delimiter_balance(code, "java"))

    java_string_vars: Set[str] = set()
    java_boxed_vars: Set[str] = set()
    java_null_vars: Dict[str, int] = {}
    java_arrays: Dict[str, Tuple[int, int]] = {}

    for m in re.finditer(r"\bString(?:\[\])?\s+([a-zA-Z_$][\w$]*)", code):
        java_string_vars.add(m.group(1))
    for m in re.finditer(r"\b(?:Integer|Long|Double|Float|Boolean|Short|Byte|Character)\s+([a-zA-Z_$][\w$]*)", code):
        java_boxed_vars.add(m.group(1))
    for m in re.finditer(r"\bint\s*\[\s*\]\s*([a-zA-Z_$][\w$]*)\s*=\s*new\s+int\s*\[\s*(\d+)\s*\]", code):
        java_arrays[m.group(1)] = (int(m.group(2)), 1)

    fn_sig_re = re.compile(
        r"^\s*(?:(?:public|private|protected|static|final|synchronized)\s+)*"
        r"([\w:<>*&\[\]]+)\s+([a-zA-Z_][\w]*)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?\{"
    )

    current_fn_name: Optional[str] = None
    current_fn_ret_type: Optional[str] = None
    last_stmt_returned = False
    current_fn_start_line = 1
    brace_depth = 0

    for idx, line in enumerate(lines, start=1):
        trimmed = line.strip()
        if not trimmed or trimmed.startswith(("//", "/*", "*")):
            continue

        sig_m = fn_sig_re.match(line)
        if sig_m and brace_depth <= 1:
            current_fn_ret_type = sig_m.group(1).strip()
            current_fn_name = sig_m.group(2).strip()
            current_fn_start_line = idx
            last_stmt_returned = False

        brace_depth += trimmed.count("{") - trimmed.count("}")

        if brace_depth == 1 and current_fn_name:
            if current_fn_ret_type and current_fn_ret_type != "void" and current_fn_name != "main":
                if not last_stmt_returned:
                    findings.append({
                        "line": current_fn_start_line,
                        "severity": "critical",
                        "rule": "JAVA-RET01",
                        "title": f"Missing return statement: non-void method '{current_fn_name}' must return a result of type {current_fn_ret_type}",
                        "fix": f"Add 'return <value>;' on all execution branches.",
                        "snippet": lines[current_fn_start_line - 1].strip(),
                    })
            current_fn_name = None

        if brace_depth == 2:
            if re.match(r"^(return\b|throw\b)", trimmed):
                last_stmt_returned = True
            elif not trimmed.startswith(("{", "}", "//", "/*", "*")):
                last_stmt_returned = False

        # Track null variables
        null_m = re.search(r"\b([a-zA-Z_$][\w$]*)\s*=\s*null\s*;", trimmed)
        if null_m:
            java_null_vars[null_m.group(1)] = idx

        for n_var, n_line in list(java_null_vars.items()):
            if n_line < idx:
                if re.search(rf"\b{re.escape(n_var)}\s*=\s*(?!null\b)", trimmed):
                    java_null_vars.pop(n_var, None)
                elif re.search(rf"\b{re.escape(n_var)}\s*\.", trimmed):
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "JAVA-NULL",
                        "title": f"Null pointer dereference risk on variable '{n_var}' (assigned null on line {n_line})",
                        "fix": f"Ensure '{n_var}' is initialized or add a null check before calling methods.",
                        "snippet": trimmed,
                    })
                    java_null_vars.pop(n_var, None)

        # String reference equality == or != check
        eq_match = re.search(r'([a-zA-Z_$][\w$]*|"[^"]*")\s*(==|!=)\s*([a-zA-Z_$][\w$]*|"[^"]*")', trimmed)
        if eq_match:
            lhs, op, rhs = eq_match.group(1), eq_match.group(2), eq_match.group(3)
            is_obj_cmp = (
                lhs.startswith('"') or rhs.startswith('"')
                or lhs in java_string_vars or rhs in java_string_vars
                or lhs in java_boxed_vars or rhs in java_boxed_vars
            )
            if is_obj_cmp and lhs != "null" and rhs != "null":
                findings.append({
                    "line": idx,
                    "severity": "critical",
                    "rule": "PMD-ES01",
                    "title": f"Reference equality ({op}) used on String/boxed object ({lhs} {op} {rhs})",
                    "fix": "Use Objects.equals(a, b) or a.equals(b) for value equality.",
                    "snippet": trimmed,
                })

        # Empty catch block
        if re.search(r"catch\s*\([^)]+\)\s*\{\s*\}", trimmed):
            findings.append({
                "line": idx,
                "severity": "warning",
                "rule": "PMD-EMPTY",
                "title": "Empty catch block silently swallows exceptions",
                "fix": "Log the caught exception or rethrow a wrapped exception.",
                "snippet": trimmed,
            })
        elif re.search(r"catch\s*\([^)]+\)\s*\{", trimmed):
            catch_body = []
            for f_line in lines[idx:]:
                f_trimmed = f_line.strip()
                if f_trimmed == "}":
                    break
                if f_trimmed and not f_trimmed.startswith(("//", "/*", "*")):
                    catch_body.append(f_trimmed)
            if not catch_body:
                findings.append({
                    "line": idx,
                    "severity": "warning",
                    "rule": "PMD-EMPTY",
                    "title": "Empty catch block silently swallows exceptions",
                    "fix": "Log the caught exception or rethrow a wrapped exception.",
                    "snippet": trimmed,
                })

        # Resource leak check: Scanner / Streams without try-with-resources
        res_m = re.search(r"\bnew\s+(Scanner|FileInputStream|FileOutputStream|BufferedReader|BufferedWriter|Socket|ServerSocket)\s*\(", trimmed)
        if res_m and "try" not in trimmed and ".close()" not in code:
            findings.append({
                "line": idx,
                "severity": "warning",
                "rule": "JAVA-RES",
                "title": f"Resource leak risk: '{res_m.group(1)}' opened without try-with-resources or .close()",
                "fix": f"Wrap {res_m.group(1)} in a try (...) block so it closes automatically.",
                "snippet": trimmed,
            })

        # Array index out of bounds
        for arr_m in re.finditer(r"\b([a-zA-Z_$][\w$]*)\s*\[\s*(\d+)\s*\]", trimmed):
            a_name = arr_m.group(1)
            a_idx = int(arr_m.group(2))
            if a_name in java_arrays and not re.match(r"^int\s*\[\s*\]", trimmed):
                cap = java_arrays[a_name][0]
                if a_idx >= cap:
                    findings.append({
                        "line": idx,
                        "severity": "critical",
                        "rule": "JAVA-BOUNDS",
                        "title": f"ArrayIndexOutOfBoundsException: Index {a_idx} out of bounds for array '{a_name}' of length {cap}",
                        "fix": f"Ensure index is between 0 and {cap - 1}.",
                        "snippet": trimmed,
                    })

        # Infinite loop: while(true) or for(;;) without exit
        if re.search(r"\b(?:while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;\s*\))", trimmed):
            sub_body = "\n".join(lines[idx - 1: min(len(lines), idx + 20)])
            if not re.search(r"\b(break|return|throw)\b", sub_body):
                findings.append({
                    "line": idx,
                    "severity": "critical",
                    "rule": "JAVA-LOOP-INF",
                    "title": "Infinite loop detected with no break, return, or exception exit",
                    "fix": "Add a termination condition or break statement inside the loop.",
                    "snippet": trimmed,
                })

        # Division by zero
        if re.search(r"[/]\s*0(?![.\d])", trimmed) or re.search(r"[%]\s*0(?![.\d])", trimmed):
            findings.append({
                "line": idx,
                "severity": "critical",
                "rule": "JAVA-ZERO",
                "title": "ArithmeticException: Division or modulo by zero literal",
                "fix": "Ensure divisor is non-zero before division.",
                "snippet": trimmed,
            })

        # Type mismatch: assigning string literal to numeric/boolean variable
        if re.match(r'^(?:int|long|short|float|double|boolean)\s+[a-zA-Z_$][\w$]*\s*=\s*"[^"]*"', trimmed):
            findings.append({
                "line": idx,
                "severity": "critical",
                "rule": "JAVA-TYPE",
                "title": "Type mismatch: cannot assign String literal to numeric/boolean variable",
                "fix": "Use a numeric/boolean literal without quotes, or declare the variable as String.",
                "snippet": trimmed,
            })

        # Missing semicolon check
        line_clean = re.sub(r"//.*$", "", trimmed).strip()
        needs_semi = (
            not line_clean.endswith((";", "{", "}", ":", ",", "\\"))
            and not re.match(r"^(if|else|for|while|switch|try|catch|public|private|protected|class|interface|enum|package|import|@)\b", line_clean)
            and not fn_sig_re.match(line_clean)
        )
        if needs_semi:
            findings.append({
                "line": idx,
                "severity": "critical",
                "rule": "JAVA-SEMI",
                "title": "Missing semicolon ';' at end of statement",
                "fix": f"Add ';' at the end of line {idx}.",
                "snippet": trimmed,
            })

    return findings


# =====================================================================
# COMPILER INTEGRATION (GCC / G++ / JAVAC)
# =====================================================================

def _run_compiler_syntax_check(code: str, lang: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    lines = code.splitlines()

    if lang in {"c", "cpp"}:
        compiler = shutil.which("g++" if lang == "cpp" else "gcc")
        if not compiler:
            return []
        suffix = ".cpp" if lang == "cpp" else ".c"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            proc = subprocess.run(
                [compiler, "-fsyntax-only", "-Wall", "-Wextra", tmp_path],
                text=True,
                capture_output=True,
                timeout=6,
            )
            for err_line in proc.stderr.splitlines():
                m = re.match(r"^.+?:(\d+):\d+:\s*(error|warning):\s*(.*)$", err_line.strip())
                if m:
                    ln = int(m.group(1))
                    kind = m.group(2)
                    msg = m.group(3)
                    findings.append({
                        "line": ln,
                        "severity": "critical" if kind == "error" else "warning",
                        "rule": f"{lang.upper()}-COMPILER" if kind == "error" else f"{lang.upper()}-WARN",
                        "title": msg,
                        "fix": "Resolve compiler diagnostic.",
                        "snippet": lines[ln - 1].strip() if 1 <= ln <= len(lines) else "",
                    })
        except Exception:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    elif lang == "java":
        javac_bin = shutil.which("javac")
        if not javac_bin:
            return []
        cls_match = re.search(r"\bpublic\s+class\s+([A-Za-z_][\w]*)", code)
        cls_name = cls_match.group(1) if cls_match else "Main"
        tmp_dir = tempfile.mkdtemp(prefix="codelens_java_")
        java_file = os.path.join(tmp_dir, f"{cls_name}.java")
        try:
            with open(java_file, "w", encoding="utf-8") as jf:
                jf.write(code)
            proc = subprocess.run(
                [javac_bin, "-Xlint:all", "-d", tmp_dir, java_file],
                text=True,
                capture_output=True,
                timeout=8,
            )
            for err_line in proc.stderr.splitlines():
                m = re.match(r"^.+?\.java:(\d+):\s*(error|warning):\s*(.*)$", err_line.strip())
                if m:
                    ln = int(m.group(1))
                    kind = m.group(2)
                    msg = m.group(3)
                    rule = "JAVA-COMPILER" if kind == "error" else "JAVA-WARN"
                    if "missing return statement" in msg.lower():
                        rule = "JAVA-RET01"
                    findings.append({
                        "line": ln,
                        "severity": "critical" if kind == "error" else "warning",
                        "rule": rule,
                        "title": msg,
                        "fix": "Resolve Java compiler diagnostic.",
                        "snippet": lines[ln - 1].strip() if 1 <= ln <= len(lines) else "",
                    })
        except Exception:
            pass
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return findings


# =====================================================================
# CYCLOMATIC COMPLEXITY
# =====================================================================

def _compute_complexity(code: str, lang: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if lang == "python" and cc_visit is not None:
        try:
            for item in cc_visit(code):
                blocks.append({
                    "name": item.name,
                    "cc": int(item.complexity),
                    "rank": cc_rank(item.complexity) if cc_rank else ("A" if item.complexity <= 5 else "B"),
                    "line": int(item.lineno),
                })
        except Exception:
            pass

    if not blocks and lizard is not None:
        ext_map = {"python": "test.py", "c": "test.c", "cpp": "test.cpp", "java": "Test.java"}
        try:
            info = lizard.analyze_file.analyze_source_code(ext_map.get(lang, "test.py"), code)
            for fn in info.function_list:
                cc = int(fn.cyclomatic_complexity)
                rank = "A" if cc <= 3 else ("B" if cc <= 6 else "C")
                blocks.append({
                    "name": fn.name,
                    "cc": cc,
                    "rank": rank,
                    "line": int(fn.start_line),
                })
        except Exception:
            pass

    if not blocks:
        branches = len(re.findall(r"\b(if|elif|for|while|case|catch)\b", code))
        cc = 1 + branches
        blocks.append({
            "name": "<module_scope>",
            "cc": cc,
            "rank": "A" if cc <= 3 else ("B" if cc <= 6 else "C"),
            "line": 1,
        })
    return blocks


# =====================================================================
# MAIN ENTRY POINT: analyze_source()
# =====================================================================

def analyze_source(code: str, language: str = "auto", filename: str = "") -> Dict[str, Any]:
    lang = detect_language(code, filename) if language == "auto" else language
    lines = code.splitlines()
    loc = len(lines)

    findings: List[Dict[str, Any]] = []

    if lang == "python":
        findings = _analyze_python_with_recovery(code)
        # Run Ruff if available
        existing_keys = {(f["line"], f["title"][:18]) for f in findings}
        for rf in _run_ruff(code):
            if (rf["line"], rf["title"][:18]) not in existing_keys:
                findings.append(rf)
                existing_keys.add((rf["line"], rf["title"][:18]))

    elif lang in {"c", "cpp"}:
        findings = _analyze_c_cpp_deep(code, lang)
        # Run GCC / G++ compiler check
        existing_keys = {(f["line"], f["rule"]) for f in findings}
        for cf in _run_compiler_syntax_check(code, lang):
            if (cf["line"], cf["rule"]) not in existing_keys:
                findings.append(cf)
                existing_keys.add((cf["line"], cf["rule"]))

    elif lang == "java":
        findings = _analyze_java_deep(code)
        # Run javac compiler check
        existing_keys = {(f["line"], f["rule"]) for f in findings}
        for jf in _run_compiler_syntax_check(code, lang):
            if (jf["line"], jf["rule"]) not in existing_keys:
                findings.append(jf)
                existing_keys.add((jf["line"], jf["rule"]))

    # Complexity
    complexity_blocks = _compute_complexity(code, lang)
    for blk in complexity_blocks:
        if blk["cc"] >= 6:
            findings.append({
                "line": blk["line"],
                "severity": "warning",
                "rule": "CC-HIGH",
                "title": f"High cyclomatic complexity (CC = {blk['cc']}) in '{blk['name']}'",
                "fix": "Refactor deeply nested branches into smaller helper functions.",
                "snippet": f"{blk['name']}() [CC={blk['cc']}, Rank={blk['rank']}]",
            })

    # Sort and deduplicate findings by (line, rule, title)
    unique_findings: List[Dict[str, Any]] = []
    seen = set()
    for f in sorted(findings, key=lambda x: x["line"]):
        key = (f["line"], f["rule"], f["title"][:25])
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    max_cc = max((b["cc"] for b in complexity_blocks), default=1)
    crit_count = sum(1 for f in unique_findings if f["severity"] == "critical")
    warn_count = sum(1 for f in unique_findings if f["severity"] == "warning")
    style_count = len(unique_findings) - crit_count - warn_count

    penalty = crit_count * 18 + warn_count * 9 + style_count * 3 + max(0, max_cc - 4) * 3
    health_score = max(5, min(100, 100 - penalty))
    heuristic_risk = "High" if (crit_count > 0 or max_cc >= 7) else ("Medium" if (warn_count > 0 or max_cc >= 4) else "Low")

    # Machine Learning prediction
    ml_score = 0.15
    ml_metrics = {}
    artifact = get_ml_artifact()
    if artifact and "pipeline" in artifact:
        try:
            prob = float(artifact["pipeline"].predict_proba([code])[0][1])
            ml_score = round(prob, 4)
            ml_metrics = artifact.get("metrics", {})
        except Exception:
            pass

    toolchains = {
        "python": "Deep Python AST + Multi-Syntax Recovery + Ruff + Radon + ML",
        "c": "Deep C Static + Memory/Pointer CFA + GCC + Lizard + ML",
        "cpp": "Deep C++ Static + RAII/Memory CFA + G++ + Lizard + ML",
        "java": "Deep Java Static + Resource/Null CFA + Javac + Lizard + ML",
    }

    return {
        "language": lang,
        "toolchain": toolchains.get(lang, toolchains["python"]),
        "loc": loc,
        "health_score": health_score,
        "max_cc": max_cc,
        "heuristic_risk": heuristic_risk,
        "ml_score": ml_score,
        "ml_metrics": ml_metrics,
        "findings": unique_findings,
        "complexity": complexity_blocks,
    }
