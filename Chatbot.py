import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables
load_dotenv()

# Setup API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=openai_api_key)

# Models and system message
MODEL = "gpt-4o-mini"
system_message = """
You are a helpful with math assistant.
Answer politely, and assist students in doing homework, submitting homework,.
"""

# --- Tools ---
def check_balance(student_id):
    print(f"Checking balance for {student_id}")
    return "$5,400"

def make_payment(student_id, amount):
    print(f"Payment of ${amount} received for student {student_id}")
    return f"Confirmed: ${amount} applied to your balance."

def request_payment_plan(student_id, proposal):
    print(f"Payment plan requested for {student_id}: {proposal}")
    return "Payment plan request received. You will be contacted shortly."

# --- Tool Descriptions ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_balance",
            "description": "Returns current tuition balance for student.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string"}
                },
                "required": ["student_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "make_payment",
            "description": "Submit a partial or full payment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string"},
                    "amount": {"type": "string"}
                },
                "required": ["student_id", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_payment_plan",
            "description": "Request a custom payment plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_id": {"type": "string"},
                    "proposal": {"type": "string"}
                },
                "required": ["student_id", "proposal"]
            }
        }
    }
]

# --- Handle Tool Calls ---
def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    if tool_call.function.name == "check_balance":
        result = check_balance(arguments["student_id"])
        return {"role": "tool", "content": result, "tool_call_id": tool_call.id}, None
    elif tool_call.function.name == "make_payment":
        result = make_payment(arguments["student_id"], arguments["amount"])
        return {"role": "tool", "content": result, "tool_call_id": tool_call.id}, None
    elif tool_call.function.name == "request_payment_plan":
        result = request_payment_plan(arguments["student_id"], arguments["proposal"])
        return {"role": "tool", "content": result, "tool_call_id": tool_call.id}, None

# --- Chat Logic ---
def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_response, _ = handle_tool_call(message)
        messages.append(message)
        messages.append(tool_response)
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages
        )

    return response.choices[0].message.content

# --- Gradio UI ---
gr.ChatInterface(fn=chat, type="messages").launch()
