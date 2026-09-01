"""Static contract tests for the optional ModelExpress RL integration."""

import ast
from pathlib import Path

_ROOT = Path(__file__).parents[3]
_MX_REFIT = _ROOT / "src" / "prime_rl" / "transports" / "weights" / "mx_refit.py"
_WEIGHTS_INIT = _ROOT / "src" / "prime_rl" / "transports" / "weights" / "__init__.py"
_CLIENTS = _ROOT / "src" / "prime_rl" / "orchestrator" / "clients.py"
_RL_ENTRYPOINT = _ROOT / "src" / "prime_rl" / "entrypoints" / "rl.py"
_MX_REVISION = "7c3a6cbdb260d48da4e002d0e1e02d754505dd9c"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return next(
        node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _call_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        parts: list[str] = []
        function = child.func
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name):
            parts.append(function.id)
        names.append(".".join(reversed(parts)))
    return names


def test_mx_refit_uses_current_transport_interfaces():
    tree = ast.parse(_MX_REFIT.read_text())

    sender = _class(tree, "MXRefitWeightSender")
    receiver = _class(tree, "MXRefitWeightReceiver")

    assert sender.bases[0].id == "WeightSender"
    assert receiver.bases[0].id == "WeightReceiver"
    assert _method(sender, "_broadcast")
    assert _method(receiver, "receive")


def test_receiver_retires_version_even_when_install_fails():
    tree = ast.parse(_MX_REFIT.read_text())
    receive = _method(_class(tree, "MXRefitWeightReceiver"), "receive")
    cleanup = next(node for node in ast.walk(receive) if isinstance(node, ast.Try))

    assert "update_weights" in _call_names(cleanup)
    assert "asyncio.to_thread" in _call_names(ast.Module(body=cleanup.finalbody, type_ignores=[]))
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "delete_weight_version"
        for node in ast.walk(ast.Module(body=cleanup.finalbody, type_ignores=[]))
    )


def test_release_wait_is_bounded():
    tree = ast.parse(_MX_REFIT.read_text())
    wait_released = _method(_class(tree, "MXRefitWeightSender"), "_wait_released")

    assert any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "TimeoutError"
        for node in ast.walk(wait_released)
    )


def test_transport_registry_constructs_both_sides():
    source = _WEIGHTS_INIT.read_text()

    assert "MXRefitWeightSender(" in source
    assert "MXRefitWeightReceiver(" in source


def test_rl_launcher_starts_modelexpress_for_refit():
    source = _RL_ENTRYPOINT.read_text()

    assert 'weight_broadcast.type in ("nixl", "mx_refit")' in source


def test_client_and_server_use_the_same_mx_revision():
    pyproject = (_ROOT / "pyproject.toml").read_text()
    installer = (_ROOT / "scripts" / "install_modelexpress.sh").read_text()

    assert f'rev = "{_MX_REVISION}"' in pyproject
    assert f'MODELEXPRESS_REF="{_MX_REVISION}"' in installer


def test_admin_update_forwards_the_version_uid():
    tree = ast.parse(_CLIENTS.read_text())
    update_weights = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "update_weights"
    )
    payload = next(
        node
        for node in ast.walk(update_weights)
        if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == "version_uid" for key in node.keys)
    )

    version_index = next(
        index for index, key in enumerate(payload.keys) if isinstance(key, ast.Constant) and key.value == "version_uid"
    )
    assert isinstance(payload.values[version_index], ast.Name)
    assert payload.values[version_index].id == "version_uid"

    gather = next(
        node
        for node in ast.walk(update_weights)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "gather"
    )
    return_exceptions = next(keyword.value for keyword in gather.keywords if keyword.arg == "return_exceptions")
    assert ast.literal_eval(return_exceptions) is True
