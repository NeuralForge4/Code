import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzer import analyze_source

passed = 0
failed = 0

def assert_findings_contain(report, expected_rule, test_name):
    global passed, failed
    rules = [f["rule"] for f in report["findings"]]
    if any(expected_rule in r for r in rules):
        print(f"  [PASS] {test_name}: found rule '{expected_rule}'")
        passed += 1
    else:
        print(f"  [FAIL] {test_name}: expected rule '{expected_rule}', but found: {rules}")
        failed += 1

print("========================================")
print("TESTING C ANALYZER")
print("========================================")

c_oob = """
#include <stdio.h>
void test_oob() {
    int arr[5];
    arr[10] = 42;
}
"""
r = analyze_source(c_oob, language="c")
assert_findings_contain(r, "CWE-787", "C Array Out-of-bounds")

c_loop_rec = """
#include <stdio.h>
void test_inf_loop() {
    while (1) {
        printf("loop");
    }
}
int test_inf_rec(int x) {
    return test_inf_rec(x);
}
"""
r = analyze_source(c_loop_rec, language="c")
assert_findings_contain(r, "C-LOOP-INF", "C Infinite Loop")
assert_findings_contain(r, "C-REC-INF", "C Infinite Recursion")

c_ret_zero = """
#include <stdio.h>
int missing_ret(int a) {
    if (a > 0) int k = a;
}
void div_zero() {
    int x = 10 / 0;
}
"""
r = analyze_source(c_ret_zero, language="c")
assert_findings_contain(r, "C-RET01", "C Missing Return Statement")
assert_findings_contain(r, "CWE-369", "C Division by Zero")

c_mem = """
#include <stdlib.h>
#include <stdio.h>
void mem_bugs() {
    int *p = (int *)malloc(sizeof(int));
    free(p);
    free(p); // double free
    *p = 10; // use after free
    int *leak = (int *)malloc(100);
}
"""
r = analyze_source(c_mem, language="c")
assert_findings_contain(r, "CWE-415", "C Double Free")
assert_findings_contain(r, "CWE-416", "C Use After Free")
assert_findings_contain(r, "CWE-401", "C Memory Leak")

c_sec = """
#include <stdio.h>
#include <string.h>
void sec_bugs(char *user_input) {
    char buf[16];
    gets(buf);
    strcpy(buf, user_input);
    printf(user_input);
}
"""
r = analyze_source(c_sec, language="c")
assert_findings_contain(r, "CWE-242", "C gets() banned function")
assert_findings_contain(r, "CWE-120", "C strcpy() buffer overflow")
assert_findings_contain(r, "CWE-134", "C Format string vulnerability")


print("\n========================================")
print("TESTING C++ ANALYZER")
print("========================================")

cpp_pairing = """
#include <iostream>
void test_pairing() {
    int* arr = new int[50];
    delete arr; // mismatch
    int* scalar = new int(10);
    delete[] scalar; // mismatch
}
"""
r = analyze_source(cpp_pairing, language="cpp")
assert_findings_contain(r, "CPP-DEL-MISMATCH", "C++ new[] with delete and new with delete[]")

cpp_dangling = """
#include <iostream>
int* dangling() {
    int local = 5;
    return &local;
}
"""
r = analyze_source(cpp_dangling, language="cpp")
assert_findings_contain(r, "CWE-562", "C++ Dangling stack pointer")

cpp_str = """
#include <iostream>
#include <string>
void str_oob() {
    std::string s = "abc";
    char c = s[10];
}
"""
r = analyze_source(cpp_str, language="cpp")
assert_findings_contain(r, "CPP-STR-BOUNDS", "C++ String index out of bounds")

cpp_stream = """
#include <fstream>
void test_stream() {
    std::ifstream file("test.txt");
    int val;
    file >> val;
}
"""
r = analyze_source(cpp_stream, language="cpp")
assert_findings_contain(r, "CPP-STREAM-CHECK", "C++ Unchecked file stream")


print("\n========================================")
print("TESTING PYTHON ANALYZER")
print("========================================")

# Multiple syntax errors + unaffected block analysis
py_multi_syntax = """
def bad_func_1(a, b)
    return a + b

def good_func_security(x):
    return eval(x)

def bad_func_2(items):
    arr = [1, 2, 3
    return items

def good_func_sqli(user):
    import sqlite3
    conn = sqlite3.connect("test.db")
    conn.execute(f"SELECT * FROM users WHERE id = '{user}'")
"""
r = analyze_source(py_multi_syntax, language="python")
print(f"  Findings in multi-syntax file: {len(r['findings'])}")
for f in r["findings"]:
    print(f"    - L{f['line']} [{f['rule']}]: {f['title']}")
assert_findings_contain(r, "PY-SYNTAX", "Python Multiple Syntax Errors (Missing colon & unclosed bracket)")
assert_findings_contain(r, "PY-SEC01", "Python Security Analysis in unaffected function (eval)")
assert_findings_contain(r, "PY-SQLI", "Python Security Analysis in unaffected function (SQL Injection)")

py_logic = """
def logic_bugs():
    items = [1, 2]
    val = items[5] # bounds
    d = {"name": "Alice"}
    k = d["non_existent_key"] # keyerror
    zero = 10 / 0 # zero div
    x = 1 + "str" # type error
    while True:
        pass # infinite loop
"""
r = analyze_source(py_logic, language="python")
assert_findings_contain(r, "PY-BOUNDS", "Python List Index Out of Range")
assert_findings_contain(r, "PY-KEYERR", "Python KeyError on Known Dict")
assert_findings_contain(r, "PY-ZERO01", "Python Division by Zero")
assert_findings_contain(r, "PY-TYPE", "Python Type Error (int + str)")
assert_findings_contain(r, "PY-LOOP01", "Python Infinite while True")

py_res = """
def leak_file():
    f = open('data.txt', 'r')
    return f.read()
"""
r = analyze_source(py_res, language="python")
assert_findings_contain(r, "PY-RES", "Python Resource Leak (open without with)")


print("\n========================================")
print("TESTING JAVA ANALYZER")
print("========================================")

java_test = """
import java.util.Scanner;
public class JavaTests {
    public boolean check(String s1, String s2) {
        if (s1 == s2) { // string reference comparison
            return true;
        }
        return false;
    }
    public void boundsAndNull() {
        int[] arr = new int[3];
        arr[10] = 5; // bounds
        String n = null;
        n.length(); // null risk
        int z = 10 / 0; // zero div
    }
    public void leak() {
        Scanner sc = new Scanner(System.in); // resource leak
    }
    public int missingReturn(int a) {
        if (a > 0) return a;
        // missing return
    }
}
"""
r = analyze_source(java_test, language="java")
assert_findings_contain(r, "PMD-ES01", "Java String Reference Comparison (==)")
assert_findings_contain(r, "JAVA-BOUNDS", "Java Array Out of Bounds")
assert_findings_contain(r, "JAVA-NULL", "Java Null Dereference Risk")
assert_findings_contain(r, "JAVA-ZERO", "Java Division by Zero")
assert_findings_contain(r, "JAVA-RES", "Java Resource Leak (Scanner)")
assert_findings_contain(r, "JAVA-RET01", "Java Missing Return Statement")

print("\n========================================")
print(f"FINAL RESULT: {passed} PASSED, {failed} FAILED")
print("========================================")
if failed > 0:
    sys.exit(1)
