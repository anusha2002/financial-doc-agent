"""
Free, local version of the agent layer -- uses Ollama's tool-calling instead of the
Anthropic API. Requires a model that supports tool use (llama3.1, qwen2.5,
mistral-nemo, firefunction-v2 all work; smaller/older models often don't).

Setup:
    ollama pull llama3.1
    pip install -r requirements-local.txt

Note: local models are noticeably less reliable at deciding *when* to call a tool
and at following the citation instruction than Claude is. Expect to iterate on the
system prompt more, and document that iteration in your README -- it's a legitimate
part of the "prompt engineering / failure modes" story for this project.
"""
import ollama
from rag import retrieve

LLM_MODEL = "llama3.1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": (
                "Search the indexed financial documents for relevant passages. "
                "Use this whenever the question requires facts, figures, or statements "
                "from the source documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a simple arithmetic expression, e.g. for computing growth rates, "
                "margins, or ratios once you have the relevant numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python-evaluable arithmetic expression, e.g. '(120-100)/100*100'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a financial research assistant with access to a document search tool and "
    "a calculator. When you use retrieve_documents, cite the source filename and page "
    "number in your final answer. When you don't have enough information, say so rather "
    "than guessing. Only use the calculator on numbers you actually retrieved or the user "
    "gave you -- never invent figures."
)


def run_tool(name: str, tool_input: dict) -> str:
    if name == "retrieve_documents":
        hits = retrieve(tool_input["query"])
        if not hits:
            return "No relevant passages found."
        return "\n\n".join(f"[{h['source']}, p.{h['page']}]: {h['text']}" for h in hits)
    elif name == "calculate":
        try:
            # NOTE: eval() is fine for a personal weekend project run locally.
            # Do not ship this pattern into anything handling untrusted input.
            result = eval(tool_input["expression"], {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"
    return f"Unknown tool: {name}"


def ask_agent(user_message: str, max_turns: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_turns):
        response = ollama.chat(model=LLM_MODEL, messages=messages, tools=TOOLS)
        message = response["message"]

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message.get("content", "")

        messages.append(message)
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            result = run_tool(name, args)
            messages.append({"role": "tool", "content": result})

    return "Reached max turns without a final answer."


if __name__ == "__main__":
    q = input("Ask the agent a question: ")
    print("\n" + ask_agent(q))
