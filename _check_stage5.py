"""Check Stage 5 syntax and imports."""
import sys, io, ast
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PASS, FAIL = "PASS", "FAIL"
results = []

def check(name, fn):
    try:
        fn()
        results.append((PASS, name))
    except Exception as e:
        results.append((FAIL, f"{name}: {e}"))

# Syntax check all stage 5 files
files = [
    "registry/__main__.py",
    "common/llm.py",
    "common/a2a_client.py",
    "common/registry_client.py",
    "customer_agent/graph.py",
    "customer_agent/agent_executor.py",
    "customer_agent/__main__.py",
    "law_agent/graph.py",
    "law_agent/agent_executor.py",
    "law_agent/__main__.py",
    "tax_agent/graph.py",
    "tax_agent/agent_executor.py",
    "tax_agent/__main__.py",
    "compliance_agent/graph.py",
    "compliance_agent/agent_executor.py",
    "compliance_agent/__main__.py",
    "test_client.py",
]

for f in files:
    def _make(path):
        def _fn():
            src = open(path, encoding="utf-8").read()
            ast.parse(src)
        return _fn
    check(f"Syntax OK: {f}", _make(f))

# Import check for core modules (no servers started)
sys.path.insert(0, ".")

def _check_a2a():
    import a2a
    from a2a.client import A2AClient
    from a2a.types import AgentCard, Message, Part, TextPart, Role
    from a2a.server.apps import A2AFastAPIApplication
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.server.tasks import TaskUpdater, InMemoryTaskStore
check("a2a-sdk: core classes importable", _check_a2a)

def _check_common():
    from common.llm import get_llm
    llm = get_llm()
    assert llm.temperature == 0.3
check("common/llm.py: get_llm() OK", _check_common)

def _check_registry():
    import importlib.util
    spec = importlib.util.spec_from_file_location("reg", "registry/__main__.py")
    mod = importlib.util.module_from_spec(spec)
    # Just parse/exec module-level without calling uvicorn.run
    # (it's guarded by if __name__ == "__main__")
    spec.loader.exec_module(mod)
    assert hasattr(mod, "app")
    assert hasattr(mod, "register")
    assert hasattr(mod, "discover")
check("registry/__main__.py: FastAPI app defined", _check_registry)

def _check_tax_graph():
    from tax_agent.graph import create_graph
    # Don't actually call get_llm() to connect — just check it's importable
    import ast
    src = open("tax_agent/graph.py", encoding="utf-8").read()
    ast.parse(src)
check("tax_agent/graph.py: create_graph importable", _check_tax_graph)

def _check_compliance_graph():
    from compliance_agent.graph import create_graph
    import ast
    src = open("compliance_agent/graph.py", encoding="utf-8").read()
    ast.parse(src)
check("compliance_agent/graph.py: create_graph importable", _check_compliance_graph)

def _check_law_graph():
    from law_agent.graph import create_graph, LawState, route_to_subagents
    from langgraph.constants import Send
    # Test routing logic without LLM
    state = {
        "question": "tax evasion",
        "context_id": "ctx1",
        "trace_id": "t1",
        "delegation_depth": 0,
        "law_analysis": "",
        "needs_tax": True,
        "needs_compliance": False,
        "tax_result": "",
        "compliance_result": "",
        "final_answer": "",
    }
    sends = route_to_subagents(state)
    nodes = [s.node for s in sends]
    assert "call_tax" in nodes, f"expected call_tax in {nodes}"
check("law_agent/graph.py: route_to_subagents dispatches correctly", _check_law_graph)

def _check_customer_graph():
    # Just verify the module is importable (graph is built on-demand)
    import importlib.util
    spec = importlib.util.spec_from_file_location("cg", "customer_agent/graph.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "build_graph")
check("customer_agent/graph.py: build_graph function defined", _check_customer_graph)

# Print results
print("\n" + "=" * 70)
print("  STAGE 5 VERIFICATION")
print("=" * 70)
passed = sum(1 for s, _ in results if s == PASS)
failed = sum(1 for s, _ in results if s == FAIL)
for status, name in results:
    icon = "✅" if status == PASS else "❌"
    print(f"  {icon}  {name}")
print("=" * 70)
print(f"  {passed}/{len(results)} checks passed  |  {failed} failed")
print("=" * 70)
