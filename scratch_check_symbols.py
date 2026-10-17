import ast
import os
import sys

def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        print(f"{filepath}:{e.lineno}: SyntaxError: {e.msg}")
        return

    # Check for undefined names or suspicious scoping
    # Let's inspect all nodes
    class ScopeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scopes = [{}]
            self.errors = []

        def visit_FunctionDef(self, node):
            # new function scope
            # compile code to check for syntax/undefined issues if any
            pass

    # We can also use python's compile(source, filepath, 'exec')
    try:
        compile(source, filepath, "exec")
    except Exception as e:
        print(f"{filepath}: compile error: {e}")

check_file("src/pipeline.py")
