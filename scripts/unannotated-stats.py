#!/usr/bin/env python3

import ast
from collections.abc import Iterator
from pathlib import Path
from types import EllipsisType
from typing import Iterable, Mapping


class ParseError(Exception):
    def __init__(self, ast: ast.AST, msg: str) -> None:
        super().__init__(f"{type(ast).__name__}: {msg}")
        self.ast = ast


class UnsupportedItemTypeError(ParseError):
    def __init__(self, ast: ast.AST) -> None:
        super().__init__(ast, "unsupported item type")


def main() -> None:
    stats: dict[Path, tuple[int, int, int]] = {}
    for stub in find_stubs():
        try:
            stats[stub] = determine_annotations(stub)
        except ParseError as exc:
            raise RuntimeError(f"{stub}:{exc.ast.lineno}:{exc}")
    # print_stats(stats)


def find_stubs() -> Iterator[Path]:
    for dir in ["stdlib", "stubs"]:
        yield from find_stubs_in_path(Path(dir))


def find_stubs_in_path(path: Path) -> Iterator[Path]:
    for p in sorted(path.iterdir()):
        if p.is_dir() and p.name != "@python2":
            yield from find_stubs_in_path(p)
        elif p.suffix == ".pyi":
            yield p


def determine_annotations(path: Path) -> tuple[int, int, int]:
    """Return number of total, partially annotated, and unannotated items in a file.

    The following count as one "item":
      * function/method argument
      * function/method return type
      * module-level and class-level fields
    """
    module = ast.parse(path.read_bytes(), path.name)
    return AnnotationCounter(module).count()


class AnnotationCounter:
    def __init__(self, module: ast.Module) -> None:
        self.module = module
        self.total = 0
        self.partial = 0
        self.unannotated = 0

    def count(self) -> tuple[int, int, int]:
        self.count_body(self.module.body)
        return self.total, self.partial, self.unannotated

    def count_body(self, body: Iterable[ast.stmt]) -> None:
        for item in body:
            match item:
                case ast.Expr():
                    self.count_expr(item)
                case ast.AnnAssign():
                    self.count_ann_assign(item)
                case ast.Assign():
                    self.count_assign(item)
                case ast.FunctionDef() | ast.AsyncFunctionDef():
                    self.count_function_def(item)
                case ast.ClassDef():
                    self.count_body(item.body)
                case ast.If():
                    self.count_body(item.body)
                case ast.Import() | ast.ImportFrom() | ast.Pass():
                    pass  # ignored ast types
                case _:
                    raise UnsupportedItemTypeError(item)

    def count_expr(self, expr: ast.Expr) -> None:
        match expr.value:
            case ast.Constant():
                if not isinstance(expr.value.value, EllipsisType):
                    raise UnsupportedItemTypeError(expr.value.value)
            case _:
                raise UnsupportedItemTypeError(expr.value)

    def count_ann_assign(self, assign: ast.AnnAssign) -> None:
        self.total += 1
        self.count_annotation(assign.annotation)

    def count_assign(self, assign: ast.Assign) -> None:
        assert len(assign.targets) == 1
        target = assign.targets[0]
        assert isinstance(target, ast.Name)
        match assign.value:
            case ast.Name() | ast.Attribute():
                pass  # type alias (ignore)
            case ast.BinOp():
                # PEP 604 union -> type alias (ignore)
                assert isinstance(assign.value.op, ast.BitOr)
            case ast.Subscript():
                # type alias (ignore)
                if not check_subscript(assign.value):
                    assert isinstance(assign.value.value, ast.Name)
                    raise ParseError(assign.value.value, f"unsupported assignment '{assign.value.value.id}'")
            case ast.Call():
                # type var (ignore)
                if not isinstance(assign.value.func, ast.Name):
                    raise UnsupportedItemTypeError(assign.value.func)
                if assign.value.func.id not in ["TypeVar", "ParamSpec", "NewType"]:
                    raise ParseError(assign.value.func, f"unsupported assignment value '{assign.value.func.id}'")
            case ast.List() | ast.Tuple():
                # __all__ (ignore)
                assert target.id == "__all__"
            case _:
                raise UnsupportedItemTypeError(assign.value)

    def count_function_def(self, func_def: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for arg in func_def.args.posonlyargs:
            self.count_argument(arg)
        for arg in func_def.args.args:
            self.count_argument(arg)
        if func_def.args.vararg:
            self.count_argument(func_def.args.vararg)
        for arg in func_def.args.kwonlyargs:
            self.count_argument(arg)
        if func_def.args.kwarg:
            self.count_argument(func_def.args.kwarg)
        self.total += 1  # return type
        self.count_annotation(func_def.returns)

    def count_argument(self, arg: ast.arg) -> None:
        self.total += 1
        self.count_annotation(arg.annotation)

    def count_annotation(self, ann: ast.expr | None) -> None:
        if ann is None:
            self.unannotated += 1
        elif not check_annotation_part(ann):
            self.partial += 1


def check_annotation_part(ann: ast.expr) -> bool:
    match ann:
        case ast.Constant():
            if ann.value is None:  # annotated with None
                return True
            raise ParseError(ann, f"unsupported constant '{ann.value}'")
        case ast.Name() | ast.Attribute():
            # TODO: _typeshed.Incomplete
            return True
        case ast.Subscript():
            return check_subscript(ann)

        case ast.BinOp():
            assert isinstance(ann.op, ast.BitOr)
            return check_annotation_part(ann.left) and check_annotation_part(ann.right)
        case _:
            raise UnsupportedItemTypeError(ann)


def attribute_names(node: ast.Name | ast.Attribute) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    assert isinstance(node.value, (ast.Name, ast.Attribute))
    return attribute_names(node.value) + [node.attr]


def normalize_name(node: ast.Name | ast.Attribute) -> str:
    # TODO: Track imports
    names = attribute_names(node)
    if len(names) == 1:
        match names[0]:
            case "tuple":
                return "tuple"
            case "Callable":
                return "collections.abc.Callable"
            case "Literal":
                return "typing.Literal"
            case _:
                return names[0]
    elif len(names) == 2:
        match names[0]:
            case "typing" | "_typing":
                return f"typing.{names[1]}"
            case _:
                return f"{names[0]}.{names[1]}"
    else:
        return ".".join(names)


def check_subscript(subscript: ast.Subscript) -> bool:
    if not isinstance(subscript.value, (ast.Name, ast.Attribute)):
        raise UnsupportedItemTypeError(subscript.value)
    match normalize_name(subscript.value):
        case "tuple":
            match subscript.slice:
                case ast.Name() | ast.Attribute() | ast.BinOp():
                    check_annotation_part(subscript.slice)
                case ast.Tuple():
                    elts = subscript.slice.elts[:]
                    if elts and is_ellipsis(elts[-1]):
                        elts = elts[:-1]
                    return all(check_annotation_part(el) for el in elts)
                case _:
                    raise UnsupportedItemTypeError(subscript.slice)
        case "collections.abc.Callable":
            assert isinstance(subscript.slice, ast.Tuple)
            assert len(subscript.slice.elts) == 2
            args, ret = subscript.slice.elts
            match args:
                case ast.Constant():
                    assert is_ellipsis(args)
                case ast.List():
                    if not all(check_annotation_part(el) for el in args.elts):
                        return False
                case ast.Name():
                    return True  # ParamSpec
                case _:
                    raise UnsupportedItemTypeError(args)
            return check_annotation_part(ret)
        case "typing.Literal":
            return True

    match subscript.slice:
        case ast.Tuple():
            return all(check_annotation_part(el) for el in subscript.slice.elts)
        case ast.expr():
            return check_annotation_part(subscript.slice)
        case _:
            raise UnsupportedItemTypeError(subscript.slice)


def is_ellipsis(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, EllipsisType)


def print_stats(stats: Mapping[Path, tuple[int, int, int]]) -> None:
    for path in sorted(stats.keys()):
        total, partial, unannotated = stats[path]
        print(f"{total:5} {partial:5} {unannotated:5} {path}")


if __name__ == "__main__":
    main()
