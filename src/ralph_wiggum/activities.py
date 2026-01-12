"""Activities for Ralph Wiggum workflow."""

import os
import anthropic
from temporalio import activity

from .models import (
    CallClaudeInput,
    CheckCompletionInput,
    GeneratePlanInput,
    ExecuteTaskInput,
)


@activity.defn
async def call_claude(input: CallClaudeInput) -> str:
    """Call Claude API with the prompt and conversation history."""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    # Build messages from history + current prompt
    messages = []

    # Add history
    for msg in input.history:
        messages.append(msg)

    # Add current user message with iteration context
    user_content = input.prompt
    if input.iteration > 0:
        user_content = f"{input.prompt}\n\n[This is iteration {input.iteration + 1}. Review your previous responses above and continue improving.]"

    messages.append({"role": "user", "content": user_content})

    # System message with iteration context
    system_msg = f"""You are working on a task iteratively. This is iteration {input.iteration + 1}.

Your task: {input.prompt}

When you have completed the task to your satisfaction, you MUST signal completion by outputting EXACTLY this tag with NO other text inside it:
<promise>COMPLETION_PHRASE</promise>

Replace COMPLETION_PHRASE with exactly this phrase: {input.completion_promise}

CRITICAL RULES FOR COMPLETION:
- The promise tag must contain ONLY the exact phrase "{input.completion_promise}" - nothing else
- Do NOT put descriptions, summaries, or explanations inside the promise tag
- WRONG: <promise>I completed the task about {input.completion_promise}</promise>
- CORRECT: <promise>{input.completion_promise}</promise>
- Only output the promise tag when you are TRULY done with the task"""

    response = client.messages.create(
        model=input.model,
        max_tokens=4096,
        system=system_msg,
        messages=messages,
    )

    # Extract text from response
    return response.content[0].text


@activity.defn
async def check_completion(input: CheckCompletionInput) -> bool:
    """Check if the response contains the completion promise."""
    promise_tag = f"<promise>{input.completion_promise}</promise>"
    return promise_tag in input.response


@activity.defn
async def generate_plan(input: GeneratePlanInput) -> list[dict]:
    """LLM generates task breakdown from prompt."""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    system = f"""Break down the user's task into distinct steps.
Output a plan using the create_plan tool.
Each step should be a single, actionable task.
Iteration: {input.iteration}"""

    if input.previous_plan:
        plan_summary = "\n".join(
            f"- {t['content']} (status: {t['status']})"
            for t in input.previous_plan
        )
        system += f"\n\nPrevious iteration's plan:\n{plan_summary}\n\nAdjust your approach based on what was tried."

    messages = list(input.history) + [{"role": "user", "content": input.prompt}]

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        system=system,
        messages=messages,
        tools=[{
            "name": "create_plan",
            "description": "Create a task plan",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                            },
                            "required": ["content"]
                        }
                    }
                },
                "required": ["tasks"]
            }
        }],
        tool_choice={"type": "tool", "name": "create_plan"}
    )

    # Extract tasks from tool call
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return [{"content": t["content"], "status": "pending"}
            for t in tool_use.input["tasks"]]


@activity.defn
async def execute_task(input: ExecuteTaskInput) -> str:
    """Execute a single task from the plan."""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    # Build context about the task
    task_context = f"Task {input.task_index + 1}/{input.total_tasks}: {input.task_content}"

    system_msg = f"""You are working on a multi-step task. This is iteration {input.iteration + 1}.

Original task: {input.original_prompt}

Current step: {task_context}

Focus on completing THIS specific step. Build on any previous work shown in the conversation history.

When you have completed ALL steps of the original task to your satisfaction, signal completion by outputting EXACTLY this tag:
<promise>{input.completion_promise}</promise>

CRITICAL: Only output the promise tag when the ENTIRE original task is complete, not just this step."""

    messages = list(input.history) + [{"role": "user", "content": task_context}]

    response = client.messages.create(
        model=input.model,
        max_tokens=4096,
        system=system_msg,
        messages=messages,
    )

    return response.content[0].text
