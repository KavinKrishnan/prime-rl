import ast
from pathlib import Path

_ROOT = Path(__file__).parents[3]


def test_watcher_uses_receiver_interval_unless_caller_overrides_it():
    path = _ROOT / "src" / "prime_rl" / "orchestrator" / "watcher.py"
    tree = ast.parse(path.read_text())
    watcher = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WeightWatcher")
    init = next(node for node in watcher.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignment = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "poll_interval"
    )

    assert ast.unparse(assignment.value) == "receiver.poll_interval if poll_interval is None else poll_interval"


def test_mx_receiver_requests_the_fast_interval():
    path = _ROOT / "src" / "prime_rl" / "transports" / "weights" / "mx_refit.py"
    tree = ast.parse(path.read_text())
    receiver = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MXRefitWeightReceiver"
    )
    assignment = next(
        node
        for node in receiver.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "poll_interval"
    )

    assert ast.literal_eval(assignment.value) == 0.1
