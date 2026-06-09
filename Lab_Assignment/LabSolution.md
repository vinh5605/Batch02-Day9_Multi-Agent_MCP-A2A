# Lab Solution — Batch02 Day9: Multi-Agent MCP / A2A

Hệ thống pháp lý đa tác nhân, xây dựng theo 5 giai đoạn từ đơn giản đến phân tán.

---

## Phần 1 — Direct LLM Calling

**File:** [`stages/stage_1_direct_llm/main.py`](../stages/stage_1_direct_llm/main.py)

Gọi LLM trực tiếp, không có tool, không có memory. LLM chỉ dựa vào dữ liệu huấn luyện.

```python
# common/llm.py — dùng chung cho cả 5 phần
def get_llm() -> ChatOpenAI:
    model   = os.getenv("LLM_MODEL", "qwen2.5:3b")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    return ChatOpenAI(model=model, openai_api_key=api_key,
                      openai_api_base=base_url, temperature=0.3)
```

```python
# stages/stage_1_direct_llm/main.py
QUESTION = "What are the legal consequences if a company breaches a non-disclosure agreement?"

async def main():
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "You are a legal expert. Provide a clear, concise analysis "
            "of the legal question asked. Keep your response under 300 words."
        )),
        HumanMessage(content=QUESTION),
    ]
    response = await llm.ainvoke(messages)
    print(response.content)
```

**Hạn chế:** Stateless, không có tools, không tra cứu được database, bị giới hạn bởi knowledge cutoff.

---

## Phần 2 — LLM + RAG & Tools

**File:** [`stages/stage_2_rag_tools/main.py`](../stages/stage_2_rag_tools/main.py)  
**Exercise:** [`exercises/exercise_2_tools.py`](../exercises/exercise_2_tools.py)

Thêm knowledge base (RAG mô phỏng) và tools. LLM quyết định dùng tool nào; ta tự viết vòng lặp tool-call.

### Knowledge Base

```python
LEGAL_KNOWLEDGE = [
    {
        "id": "nda_trade_secret",
        "keywords": ["nda", "non-disclosure", "confidential", "trade secret"],
        "text": (
            "NDA breaches may trigger both contractual and statutory liability. "
            "Under the DTSA (18 U.S.C. § 1836): injunctive relief, actual damages + "
            "unjust enrichment, exemplary damages up to 2x for willful misappropriation, "
            "and attorney's fees. Criminal prosecution possible under Economic Espionage "
            "Act (18 U.S.C. § 1832) with penalties up to $5M."
        ),
    },
    # ... thêm các entries khác (ucc_breach, dtsa_details, liquidated_damages, injunctive_relief)
]
```

### Tools

```python
@tool
def search_legal_database(query: str) -> str:
    """Search the legal knowledge base for relevant statutes and case law."""
    query_words = set(query.lower().split())
    scored = [(len(query_words & set(e["keywords"])), e)
              for e in LEGAL_KNOWLEDGE if query_words & set(e["keywords"])]
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join(f"[{e['id']}] {e['text']}" for _, e in scored[:2]) \
           or "No relevant legal sources found."

@tool
def calculate_damages(breach_type: str, contract_value: float) -> str:
    """Calculate estimated damages for a contract breach."""
    multiplier = 2.0 if "willful" in breach_type.lower() else \
                 1.0 if "negligent" in breach_type.lower() else 1.5
    base = contract_value * multiplier
    fees = contract_value * 0.15
    return (f"Damages: ${base:,.2f} + attorney fees ${fees:,.2f} "
            f"= total ${base + fees:,.2f}")
```

### Vòng lặp tool-call thủ công

```python
async def main():
    llm = get_llm()
    llm_with_tools = llm.bind_tools([search_legal_database, calculate_damages])
    tool_map = {t.name: t for t in [search_legal_database, calculate_damages]}

    messages = [SystemMessage(content="..."), HumanMessage(content=QUESTION)]

    # Step 1: LLM quyết định gọi tool nào
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)

    # Step 2: Thực thi từng tool
    for tc in response.tool_calls:
        result = await tool_map[tc["name"]].ainvoke(tc["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Step 3: LLM tổng hợp câu trả lời cuối
    final = await llm_with_tools.ainvoke(messages)
    print(final.content)
```

### Exercise 2 (đã hoàn thành)

```python
# exercises/exercise_2_tools.py
# Thêm entry luật lao động vào LEGAL_KNOWLEDGE
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể đơn phương "
        "chấm dứt hợp đồng trong các trường hợp: (1) người lao động thường xuyên không "
        "hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị 12 tháng chưa khỏi; "
        "(3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}

# Tool kiểm tra thời hiệu khởi kiện
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort":     "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

---

## Phần 3 — Single Agent (ReAct Loop)

**File:** [`stages/stage_3_single_agent/main.py`](../stages/stage_3_single_agent/main.py)

Dùng `create_react_agent` của LangGraph để tự động hóa vòng lặp **Think → Act → Observe**.

```python
TOOLS = [search_legal_database, calculate_penalty, check_compliance_requirements]

SYSTEM_PROMPT = (
    "You are a legal analyst agent with access to tools for searching legal databases, "
    "calculating penalties, and checking compliance requirements. Use these tools to build "
    "a comprehensive analysis. Search for each legal area separately."
)

async def main():
    from langgraph.prebuilt import create_react_agent

    llm   = get_llm()
    graph = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)

    inputs = {"messages": [{"role": "user", "content": QUESTION}]}

    async for chunk in graph.astream(inputs, stream_mode="updates"):
        for node_name, update in chunk.items():
            for msg in update.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"[THINK+ACT] Tool: {[tc['name'] for tc in msg.tool_calls]}")
                elif msg.type == "tool":
                    print(f"[OBSERVE] {msg.content[:200]}")
                elif msg.type == "ai" and msg.content:
                    print(f"[FINAL ANSWER]\n{msg.content}")
```

**Tools thêm so với Phần 2:**

```python
@tool
def calculate_penalty(violation_type: str, severity: str, annual_revenue: float) -> str:
    """Calculate estimated legal penalties."""
    multipliers = {"low": 0.01, "medium": 0.05, "high": 0.10}
    base = annual_revenue * multipliers.get(severity.lower(), 0.05)
    extras = {
        "tax":      "Plus potential criminal charges and 75% civil fraud penalty.",
        "privacy":  "Plus GDPR fines up to 4% of global revenue.",
        "contract": "Plus consequential damages and attorney's fees.",
    }
    extra = next((v for k, v in extras.items() if k in violation_type.lower()),
                 "Additional regulatory sanctions may apply.")
    return f"Base penalty: ${base:,.2f} (revenue: ${annual_revenue:,.2f}). {extra}"

@tool
def check_compliance_requirements(industry: str, company_size: str) -> str:
    """Check which regulatory frameworks apply."""
    frameworks = {
        "technology": ["CCPA/CPRA", "GDPR", "FTC Act Section 5", "SOC 2"],
        "finance":    ["SOX", "BSA/AML", "Dodd-Frank", "SEC Regulations", "FCPA"],
        "healthcare": ["HIPAA", "HITECH Act", "FTC Health Breach Notification"],
    }
    applicable = frameworks.get(industry.lower(), ["FTC Act Section 5"])
    return f"Frameworks for {industry} ({company_size}): {', '.join(applicable)}"
```

---

## Phần 4 — Multi-Agent In-Process

**File:** [`stages/stage_4_milti_agent/main.py`](../stages/stage_4_milti_agent/main.py)  
**Exercise:** [`exercises/exercise_4_multiagent.py`](../exercises/exercise_4_multiagent.py)

Nhiều agent chuyên biệt cùng làm việc trong một process, sử dụng **LangGraph StateGraph** và **Send API** để chạy song song.

### State

```python
from typing import Annotated, TypedDict
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

def _last_wins(a: str, b: str) -> str:
    return b if b else a

class LegalState(TypedDict):
    question:           str
    law_analysis:       str
    needs_tax:          bool
    needs_compliance:   bool
    tax_result:         Annotated[str, _last_wins]    # parallel-safe
    compliance_result:  Annotated[str, _last_wins]    # parallel-safe
    final_answer:       str
```

### Các nodes

```python
async def analyze_law(state: LegalState) -> dict:
    """Lead attorney phân tích khía cạnh pháp lý tổng quát."""
    llm = get_llm()
    result = await llm.ainvoke([
        SystemMessage(content="You are a senior corporate litigation attorney..."),
        HumanMessage(content=state["question"]),
    ])
    return {"law_analysis": result.content}

async def check_routing(state: LegalState) -> dict:
    """Router dùng LLM để quyết định cần specialist nào."""
    llm = get_llm()
    result = await llm.ainvoke([
        SystemMessage(content=(
            'Reply with ONLY valid JSON:\n'
            '{"needs_tax": <true|false>, "needs_compliance": <true|false>}'
        )),
        HumanMessage(content=state["question"]),
    ])
    parsed = json.loads(result.content.strip())
    return {"needs_tax": parsed.get("needs_tax", True),
            "needs_compliance": parsed.get("needs_compliance", True)}

def route_to_specialists(state: LegalState) -> list[Send]:
    """Routing function: trả về Send objects để LangGraph chạy song song."""
    sends = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax_specialist", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance_specialist", state))
    return sends or [Send("aggregate", state)]

async def call_tax_specialist(state: LegalState) -> dict:
    """Tax specialist — ReAct agent inline."""
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(
        model=get_llm(), tools=[search_tax_law],
        prompt="You are a specialist tax attorney and CPA..."
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    return {"tax_result": result["messages"][-1].content}

async def call_compliance_specialist(state: LegalState) -> dict:
    """Compliance specialist — ReAct agent inline."""
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(
        model=get_llm(), tools=[search_compliance_law],
        prompt="You are a senior regulatory compliance officer..."
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    return {"compliance_result": result["messages"][-1].content}

async def aggregate(state: LegalState) -> dict:
    """Tổng hợp tất cả phân tích thành câu trả lời cuối."""
    sections = []
    if state.get("law_analysis"):
        sections.append(f"## Legal Analysis\n{state['law_analysis']}")
    if state.get("tax_result"):
        sections.append(f"## Tax Analysis\n{state['tax_result']}")
    if state.get("compliance_result"):
        sections.append(f"## Regulatory Compliance\n{state['compliance_result']}")
    result = await get_llm().ainvoke([
        SystemMessage(content="Synthesise into a cohesive, well-structured response..."),
        HumanMessage(content="\n\n---\n\n".join(sections)),
    ])
    return {"final_answer": result.content}
```

### Graph construction

```python
def create_graph():
    graph = StateGraph(LegalState)
    graph.add_node("analyze_law",              analyze_law)
    graph.add_node("check_routing",            check_routing)
    graph.add_node("call_tax_specialist",      call_tax_specialist)
    graph.add_node("call_compliance_specialist", call_compliance_specialist)
    graph.add_node("aggregate",                aggregate)

    graph.set_entry_point("analyze_law")
    graph.add_edge("analyze_law", "check_routing")
    graph.add_conditional_edges(
        "check_routing", route_to_specialists,
        ["call_tax_specialist", "call_compliance_specialist", "aggregate"],
    )
    graph.add_edge("call_tax_specialist",       "aggregate")
    graph.add_edge("call_compliance_specialist","aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()
```

### Exercise 4 — Privacy Agent (đã hoàn thành)

```python
# exercises/exercise_4_multiagent.py

class State(TypedDict):
    question:            str
    law_analysis:        Annotated[str, _last_wins]
    tax_analysis:        Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis:    Annotated[str, _last_wins]   # field mới
    final_response:      str

def privacy_agent(state: State) -> dict:
    """Agent chuyên về GDPR và bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
Tập trung: GDPR, CCPA, data protection, privacy rights, data breach notification."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}

def check_routing(state: State) -> list[Send]:
    """Conditional routing: chỉ gọi agent khi câu hỏi có từ khóa liên quan."""
    q = state["question"].lower()
    tasks = []
    if any(kw in q for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    if any(kw in q for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    if any(kw in q for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    return tasks or [Send("aggregate_results", state)]
```

---

## Phần 5 — Distributed A2A System

**Agents:** `customer_agent/`, `law_agent/`, `tax_agent/`, `compliance_agent/`, `registry/`

Mỗi agent là một service HTTP độc lập. Giao tiếp qua A2A protocol. Registry trung tâm để discover agents.

### Kiến trúc

```
Registry (10000)  ← agents tự đăng ký khi khởi động
     ↓
Customer Agent (10100)  →  Law Agent (10101)
                                ↓
                     ┌──────────┴──────────┐
                     ↓                     ↓
           Tax Agent (10102)   Compliance Agent (10103)
```

### Registry Client

```python
# common/registry_client.py
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:10000")

async def discover(task: str) -> str:
    """Tìm endpoint của agent xử lý task."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{REGISTRY_URL}/discover/{task}")
        resp.raise_for_status()
        return resp.json()["endpoint"]

async def register(agent_info: dict) -> None:
    """Agent tự đăng ký vào registry khi khởi động."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{REGISTRY_URL}/register", json=agent_info)
        resp.raise_for_status()
```

### Customer Agent — điểm vào

```python
# customer_agent/graph.py
CUSTOMER_SYSTEM_PROMPT = """You are a helpful legal assistant at the front desk.
Use the `delegate_to_legal_agent` tool for any substantive legal question."""

def build_graph(trace_id: str, context_id: str, depth: int):
    @tool
    async def delegate_to_legal_agent(question: str) -> str:
        """Send a legal question to the Law Agent for comprehensive analysis."""
        endpoint = await discover("legal_question")
        return await delegate(
            endpoint=endpoint, question=question,
            context_id=context_id, trace_id=trace_id, depth=depth + 1,
        )

    return create_react_agent(
        model=get_llm(),
        tools=[delegate_to_legal_agent],
        prompt=CUSTOMER_SYSTEM_PROMPT,
    )
```

### Law Agent — orchestrator

```python
# law_agent/graph.py
class LawState(TypedDict):
    question:           str
    context_id:         str
    trace_id:           str
    delegation_depth:   int
    law_analysis:       str
    needs_tax:          bool
    needs_compliance:   bool
    tax_result:         Annotated[str, _last_wins]
    compliance_result:  Annotated[str, _last_wins]
    final_answer:       str

async def call_tax(state: LawState) -> dict:
    """Delegate đến Tax Agent qua A2A HTTP."""
    endpoint = await discover("tax_question")
    result = await delegate(
        endpoint=endpoint, question=state["question"],
        context_id=state["context_id"], trace_id=state["trace_id"],
        depth=state.get("delegation_depth", 0) + 1,
    )
    return {"tax_result": result}

async def call_compliance(state: LawState) -> dict:
    """Delegate đến Compliance Agent qua A2A HTTP."""
    endpoint = await discover("compliance_question")
    result = await delegate(
        endpoint=endpoint, question=state["question"],
        context_id=state["context_id"], trace_id=state["trace_id"],
        depth=state.get("delegation_depth", 0) + 1,
    )
    return {"compliance_result": result}
```

### Tax Agent — specialist

```python
# tax_agent/graph.py
TAX_SYSTEM_PROMPT = """You are a specialist tax attorney and CPA with expertise in:
- Tax evasion vs. avoidance — legal distinctions and consequences
- IRS enforcement, audits, and criminal referrals
- Penalties under IRC §§ 6651, 6662, 6663
- FBAR/FATCA requirements for offshore accounts
- Tax fraud statutes (18 U.S.C. § 7201 – § 7207)
..."""

def create_graph():
    return create_react_agent(model=get_llm(), tools=[], prompt=TAX_SYSTEM_PROMPT)
```

### Compliance Agent — specialist

```python
# compliance_agent/graph.py
COMPLIANCE_SYSTEM_PROMPT = """You are a senior regulatory compliance officer with expertise in:
- SEC enforcement and securities law violations
- SOX (Sarbanes-Oxley) compliance
- FCPA anti-bribery provisions
- AML/BSA requirements
- GDPR, CCPA, and data privacy compliance
..."""

def create_graph():
    return create_react_agent(model=get_llm(), tools=[], prompt=COMPLIANCE_SYSTEM_PROMPT)
```

### AgentExecutor (bridge A2A ↔ LangGraph)

```python
# law_agent/agent_executor.py
class LawAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question   = self._extract_question(context)
        context_id = context.context_id or str(uuid4())
        trace_id   = (context.message.metadata or {}).get("trace_id", str(uuid4()))
        depth      = int((context.message.metadata or {}).get("delegation_depth", 0))

        updater = TaskUpdater(event_queue, context.task_id, context_id)
        await updater.submit()
        await updater.start_work()

        result = await _graph.ainvoke({
            "question": question, "context_id": context_id,
            "trace_id": trace_id, "delegation_depth": depth,
            # ... các fields khác khởi tạo rỗng
        })

        answer = result.get("final_answer") or result.get("law_analysis", "")
        await updater.add_artifact(parts=[Part(root=TextPart(text=answer))],
                                   name="legal_analysis")
        await updater.complete()
```

### Khởi động toàn bộ hệ thống

```bash
./start_all.sh   # khởi động 5 services: registry + 4 agents
```

---

## So Sánh 5 Giai Đoạn

| Giai đoạn | Pattern | Đặc điểm chính | Độ phức tạp |
|---|---|---|---|
| 1 | Direct LLM | Gọi LLM trực tiếp, không tool | ⭐ |
| 2 | LLM + Tools | Bind tools, tool-call loop thủ công | ⭐⭐ |
| 3 | ReAct Agent | `create_react_agent`, autonomous loop | ⭐⭐⭐ |
| 4 | Multi-Agent In-Process | StateGraph + Send API, parallel | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | HTTP services, registry, A2A protocol | ⭐⭐⭐⭐⭐ |
