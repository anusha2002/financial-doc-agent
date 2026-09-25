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
    "gave you -- never invent figures.\n\n"
    "For questions involving more than one company, time period, or comparison: retrieve "
    "information for EACH entity or period separately, using a distinct, specific "
    "retrieve_documents call for each one. Do not answer a comparison using data for only "
    "one side of it.\n\n"
    "If a question asks for a rate, growth percentage, or margin that is not explicitly "
    "stated in the documents but CAN be computed from numbers you retrieved (e.g. revenue "
    "and cost of revenue), use the calculate tool to compute it. Do not say the information "
    "is unavailable if the underlying numbers were actually retrieved.\n\n"
    "If a question asks you to list multiple items (e.g. all business segments, all risk "
    "factors) and your first retrieval seems incomplete or partial, retrieve again with a "
    "more specific query before finalizing your answer."
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