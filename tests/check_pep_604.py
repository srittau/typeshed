#!/usr/bin/env python3

import ast
import sys
from pathlib import Path

typeshed_path = Path(__file__).parent.parent
sys.path.append(str(typeshed_path / "src"))

from typeshed_utils import PY2_PATH, find_stubs_in_paths, stdlib_path, stubs_path  # noqa E402


def check_pep_604(tree: ast.AST, path: Path) -> list[str]:
    errors = []

    class UnionFinder(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name) and node.value.id == "Union" and isinstance(node.slice, ast.Tuple):
                new_syntax = " | ".join(ast.unparse(x) for x in node.slice.elts)
                errors.append((f"{path}:{node.lineno}: Use PEP 604 syntax for Union, e.g. `{new_syntax}`"))
            if isinstance(node.value, ast.Name) and node.value.id == "Optional":
                new_syntax = f"{ast.unparse(node.slice)} | None"
                errors.append((f"{path}:{node.lineno}: Use PEP 604 syntax for Optional, e.g. `{new_syntax}`"))

    # This doesn't check type aliases (or type var bounds, etc), since those are not
    # currently supported
    class AnnotationFinder(ast.NodeVisitor):
        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            UnionFinder().visit(node.annotation)

        def visit_arg(self, node: ast.arg) -> None:
            if node.annotation is not None:
                UnionFinder().visit(node.annotation)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.returns is not None:
                UnionFinder().visit(node.returns)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node.returns is not None:
                UnionFinder().visit(node.returns)
            self.generic_visit(node)

    AnnotationFinder().visit(tree)
    return errors


def main() -> None:
    errors: list[str] = []
    for path in find_stubs_in_paths([stdlib_path(typeshed_path), stubs_path(typeshed_path)]):
        if PY2_PATH in path.parts:
            continue
        if "protobuf" in path.parts:  # TODO: fix protobuf stubs
            continue

        with open(path) as f:
            tree = ast.parse(f.read())
        errors.extend(check_pep_604(tree, path))

    if errors:
        print("\n".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
