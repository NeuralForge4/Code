import os
import random
import pandas as pd

random.seed(42)
samples = []

def add_sample(code: str, label: int, category: str, lang: str):
    code_clean = code.strip()
    if code_clean:
        samples.append({
            "code": code_clean + "\n",
            "label": int(label),
            "category": category,
            "language": lang
        })

# ----------------- CLEAN SAMPLES GENERATION -----------------
# 1. Python Clean
py_algorithms = [
    ("bubble_sort", """def bubble_sort_{v}(items: list[int]) -> list[int]:
    n = len(items)
    res = list(items)
    for i in range(n):
        for j in range(0, n - i - 1):
            if res[j] > res[j + 1]:
                res[j], res[j + 1] = res[j + 1], res[j]
    return res"""),
    ("find_median", """def find_median_{v}(numbers: list[float]) -> float:
    if not numbers:
        return 0.0
    s = sorted(numbers)
    mid = len(s) // 2
    if len(s) % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return float(s[mid])"""),
    ("lru_cache_node", """class CacheEntry_{v}:
    def __init__(self, key: str, val: int) -> None:
        self.key = key
        self.val = val
        self.hits = 0
    def touch(self) -> None:
        self.hits += 1"""),
    ("json_loader", """import json
def safe_parse_payload_{v}(raw_str: str) -> dict:
    try:
        return json.loads(raw_str)
    except (json.JSONDecodeError, TypeError):
        return {}"""),
    ("matrix_transpose", """def matrix_transpose_{v}(mat: list[list[int]]) -> list[list[int]]:
    if not mat or not mat[0]:
        return []
    rows, cols = len(mat), len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]"""),
    ("regex_validator", """import re
def validate_email_{v}(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))""")
]

for name, tmpl in py_algorithms:
    for v in range(12):
        add_sample(tmpl.replace("{v}", str(v)), 0, "clean", "python")

# 2. C Clean
c_clean_patterns = [
    ("array_sum", """#include <stdio.h>
int sum_array_{v}(const int *arr, int len) {
    if (arr == NULL || len <= 0) return 0;
    int acc = 0;
    for (int i = 0; i < len; ++i) {
        acc += arr[i];
    }
    return acc;
}"""),
    ("string_len", """#include <stddef.h>
size_t safe_strlen_{v}(const char *s) {
    if (s == NULL) return 0;
    size_t count = 0;
    while (s[count] != '\\0') {
        count++;
    }
    return count;
}"""),
    ("alloc_free", """#include <stdlib.h>
int* allocate_buffer_{v}(size_t count) {
    if (count == 0) return NULL;
    int *buf = (int *)malloc(count * sizeof(int));
    if (buf == NULL) return NULL;
    for (size_t i = 0; i < count; ++i) buf[i] = (int)i;
    return buf;
}
void release_buffer_{v}(int *buf) {
    if (buf != NULL) free(buf);
}"""),
    ("swap_values", """#include <stdio.h>
void swap_integers_{v}(int *x, int *y) {
    if (x == NULL || y == NULL) return;
    int temp = *x;
    *x = *y;
    *y = temp;
}""")
]

for name, tmpl in c_clean_patterns:
    for v in range(12):
        add_sample(tmpl.replace("{v}", str(v)), 0, "clean", "c")

# 3. C++ Clean
cpp_clean_patterns = [
    ("vector_sum", """#include <iostream>
#include <vector>
#include <numeric>
int compute_vector_sum_{v}(const std::vector<int>& data) {
    return std::accumulate(data.begin(), data.end(), 0);
}"""),
    ("unique_ptr_factory", """#include <memory>
#include <string>
std::unique_ptr<std::string> make_label_{v}(const std::string& prefix) {
    return std::make_unique<std::string>(prefix + "_{v}");
}"""),
    ("file_stream_safe", """#include <fstream>
#include <string>
bool write_data_{v}(const std::string& filename, const std::string& content) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;
    out << content;
    out.close();
    return true;
}"""),
    ("oop_point", """#include <cmath>
class Point2D_{v} {
private:
    double x_, y_;
public:
    Point2D_{v}(double x, double y) : x_(x), y_(y) {}
    double distance() const { return std::sqrt(x_ * x_ + y_ * y_); }
};""")
]

for name, tmpl in cpp_clean_patterns:
    for v in range(12):
        add_sample(tmpl.replace("{v}", str(v)), 0, "clean", "cpp")

# 4. Java Clean
java_clean_patterns = [
    ("math_service", """import java.util.Objects;
public class MathService_{v} {
    public int add(int a, int b) {
        return a + b;
    }
    public boolean checkEquality(String first, String second) {
        return Objects.equals(first, second);
    }
}"""),
    ("try_with_resources", """import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
public class SafeFileReader_{v} {
    public static String getHeader(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            return reader.readLine();
        }
    }
}"""),
    ("list_processor", """import java.util.List;
import java.util.ArrayList;
public class ListFilter_{v} {
    public static List<Integer> filterPositive(List<Integer> src) {
        if (src == null) return new ArrayList<>();
        List<Integer> out = new ArrayList<>();
        for (Integer num : src) {
            if (num != null && num > 0) out.add(num);
        }
        return out;
    }
}"""),
    ("account_model", """public class BankAccount_{v} {
    private double balance;
    public BankAccount_{v}(double initial) {
        this.balance = Math.max(0.0, initial);
    }
    public void deposit(double amount) {
        if (amount > 0) balance += amount;
    }
    public double getBalance() {
        return balance;
    }
}""")
]

for name, tmpl in java_clean_patterns:
    for v in range(12):
        add_sample(tmpl.replace("{v}", str(v)), 0, "clean", "java")


# ----------------- DEFECTIVE SAMPLES GENERATION -----------------
# Memory Safety (C and C++)
for v in range(16):
    add_sample(f"""#include <stdlib.h>
void leak_test_{v}() {{
    int *data = (int *)malloc(sizeof(int) * {v + 10});
    data[0] = {v};
    /* missing free */
}}""", 1, "memory_safety", "c")

    add_sample(f"""#include <stdlib.h>
void double_free_{v}() {{
    char *buf = (char *)malloc(64);
    if (!buf) return;
    free(buf);
    free(buf);
}}""", 1, "memory_safety", "c")

    add_sample(f"""#include <stdlib.h>
#include <stdio.h>
void uaf_{v}() {{
    int *p = (int *)malloc(sizeof(int));
    *p = {v};
    free(p);
    printf("%d\\n", *p);
}}""", 1, "memory_safety", "c")

    add_sample(f"""#include <iostream>
void new_mismatch_{v}() {{
    int* array = new int[{v + 10}];
    array[0] = 100;
    delete array; // BUG: delete instead of delete[]
}}""", 1, "memory_safety", "cpp")

    add_sample(f"""#include <iostream>
void scalar_mismatch_{v}() {{
    double* val = new double(3.14);
    delete[] val; // BUG: delete[] instead of delete
}}""", 1, "memory_safety", "cpp")

    add_sample(f"""#include <iostream>
const int& dangling_ref_{v}() {{
    int stack_num = {v} * 10;
    return stack_num; // BUG: reference to stack variable
}}""", 1, "memory_safety", "cpp")

# Security Vulnerabilities (Python, C, Java)
for v in range(14):
    add_sample(f"""def eval_call_{v}(cmd_input: str):
    return eval(cmd_input)""", 1, "security", "python")

    add_sample(f"""import os
def os_sys_{v}(user_arg: str):
    os.system("echo " + user_arg)""", 1, "security", "python")

    add_sample(f"""import sqlite3
def sql_inject_{v}(uid: str):
    conn = sqlite3.connect("db_{v}.sqlite")
    conn.execute(f"SELECT * FROM users WHERE id = '{{uid}}'")""", 1, "security", "python")

    add_sample(f"""#include <stdio.h>
void gets_call_{v}() {{
    char buf[32];
    gets(buf);
}}""", 1, "security", "c")

    add_sample(f"""#include <string.h>
void strcpy_overflow_{v}(char *src) {{
    char dest[16];
    strcpy(dest, src);
}}""", 1, "security", "c")

    add_sample(f"""#include <stdio.h>
void format_string_{v}(char *fmt) {{
    printf(fmt);
}}""", 1, "security", "c")

    add_sample(f"""public class StringRefEq_{v} {{
    public boolean check(String a, String b) {{
        return a == b; // BUG: Reference equality
    }}
}}""", 1, "security", "java")

# Resource Leaks (Python, C++, Java)
for v in range(12):
    add_sample(f"""def file_unclosed_{v}(filepath: str):
    f = open(filepath, 'r')
    content = f.read()
    return content""", 1, "resource_leak", "python")

    add_sample(f"""import java.util.Scanner;
import java.io.File;
public class ScannerLeak_{v} {{
    public void scan(String p) throws Exception {{
        Scanner sc = new Scanner(new File(p));
        while (sc.hasNext()) System.out.println(sc.next());
    }}
}}""", 1, "resource_leak", "java")

    add_sample(f"""import java.io.FileInputStream;
public class StreamLeak_{v} {{
    public void readBytes(String path) throws Exception {{
        FileInputStream fis = new FileInputStream(path);
        int b = fis.read();
    }}
}}""", 1, "resource_leak", "java")

# Null or Bounds (Python, C, C++, Java)
for v in range(12):
    add_sample(f"""def list_oob_{v}():
    items = [1, 2, 3]
    return items[{v + 10}]""", 1, "null_or_bounds", "python")

    add_sample(f"""#include <stdio.h>
void static_bounds_{v}() {{
    int arr[5];
    arr[{v + 6}] = 99;
}}""", 1, "null_or_bounds", "c")

    add_sample(f"""#include <iostream>
#include <string>
void str_bounds_{v}() {{
    std::string text = "test";
    char c = text[{v + 10}];
}}""", 1, "null_or_bounds", "cpp")

    add_sample(f"""public class NullDeref_{v} {{
    public int getLength() {{
        String str = null;
        return str.length();
    }}
}}""", 1, "null_or_bounds", "java")

    add_sample(f"""public class JavaBounds_{v} {{
    public int access() {{
        int[] vals = new int[3];
        return vals[{v + 5}];
    }}
}}""", 1, "null_or_bounds", "java")

# Control Flow & Recursion (Python, C, C++, Java)
for v in range(12):
    add_sample(f"""def loop_forever_{v}():
    while True:
        val = {v}""", 1, "control_flow", "python")

    add_sample(f"""def recurse_forever_{v}(x: int) -> int:
    return recurse_forever_{v}(x)""", 1, "control_flow", "python")

    add_sample(f"""#include <stdio.h>
void infinite_while_{v}() {{
    while (1) {{
        int x = {v};
    }}
}}""", 1, "control_flow", "c")

    add_sample(f"""#include <stdio.h>
int missing_ret_{v}(int x) {{
    if (x > 0) return x;
}}""", 1, "control_flow", "c")

    add_sample(f"""public class InfiniteJava_{v} {{
    public void run() {{
        while (true) {{
            int k = {v};
        }}
    }}
}}""", 1, "control_flow", "java")

# Syntax Errors (Python, C)
for v in range(12):
    add_sample(f"""def syntax_func_{v}(a, b)
    return a + b + {v}""", 1, "syntax_error", "python")

    add_sample(f"""def unclosed_parens_{v}(x):
    total = (x * 2 + ({v} + 1)
    return total""", 1, "syntax_error", "python")

    add_sample(f"""#include <stdio.h>
void no_semicolon_{v}() {{
    int num = {v}
    printf("%d\\n", num);
}}""", 1, "syntax_error", "c")

df = pd.DataFrame(samples).drop_duplicates(subset=["code"]).reset_index(drop=True)

out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "labeled_code_dataset.csv")
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} completely unique labeled code snippets.")
print("Category distribution:")
print(df["category"].value_counts())
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nLanguage distribution:")
print(df["language"].value_counts())
