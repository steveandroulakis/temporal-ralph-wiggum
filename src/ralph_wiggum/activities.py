"""Activities for Ralph Wiggum workflow."""

import os
import anthropic
from temporalio import activity

from .models import (
    DEFAULT_MODEL,
    DecideIterationInput,
    DecideIterationOutput,
    GenerateTasksInput,
    ExecuteTaskInput,
    EvaluateIterationInput,
    EvaluateIterationOutput,
)


@activity.defn
async def decide_iteration_mode(input: DecideIterationInput) -> DecideIterationOutput:
    """Decide whether this iteration should be single-task or multi-task."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    history_context = ""
    if input.history:
        history_context = "\n\nRecent conversation:\n" + "\n".join(
            f"[{m['role']}]: {m['content'][:500]}..." if len(m['content']) > 500 else f"[{m['role']}]: {m['content']}"
            for m in input.history[-3:]
        )

    system = f"""You are deciding how to approach the next iteration of work.

Original task: {input.prompt}

{f"Progress so far: {input.progress_summary}" if input.progress_summary else "No progress yet."}
{history_context}

Iteration: {input.iteration + 1}

Decide between:
- "single": One focused task for this iteration (simpler, more focused)
- "multi": Multiple related tasks for this iteration (when several steps needed)

Use the decide_mode tool to output your decision.

CRITICAL RULES:
- DO NOT include tasks about signaling completion or emitting tags
- Focus on actual work that advances the goal
- PREFER "multi" mode to break work into granular steps when possible
- Only use "single" when the remaining work is truly atomic/indivisible
- Err on side of smaller, focused tasks over bundling related work
"""

    messages = [{"role": "user", "content": "Decide the approach for the next iteration."}]

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
        tools=[
            {
                "name": "decide_mode",
                "description": "Decide iteration mode",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["single", "multi"],
                            "description": "Whether to do a single task or multiple tasks",
                        },
                        "single_task_content": {
                            "type": "string",
                            "description": "If mode=single, the task content. Empty if mode=multi.",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Brief explanation of why this mode was chosen",
                        },
                    },
                    "required": ["mode", "rationale"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "decide_mode"},
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    result = tool_use.input

    return DecideIterationOutput(
        mode=result["mode"],
        single_task_content=result.get("single_task_content", ""),
        rationale=result["rationale"],
    )


@activity.defn
async def generate_tasks(input: GenerateTasksInput) -> list[dict]:
    """Generate 2-5 tasks for current iteration (multi-task mode)."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    history_context = ""
    if input.history:
        history_context = "\n\nRecent conversation:\n" + "\n".join(
            f"[{m['role']}]: {m['content'][:500]}..." if len(m['content']) > 500 else f"[{m['role']}]: {m['content']}"
            for m in input.history[-3:]
        )

    system = f"""Generate 2-5 concrete tasks for THIS ITERATION ONLY.

Original task: {input.prompt}

{f"Progress so far: {input.progress_summary}" if input.progress_summary else "No progress yet."}
{history_context}

Iteration: {input.iteration + 1}

CRITICAL - YOU ARE IN A LOOP:
- This workflow runs in a loop until completion is detected
- If this iteration doesn't complete the goal, another iteration runs automatically with your progress
- DO NOT generate "revision" or "improvement" tasks - the loop handles re-iteration naturally
- Generate tasks for the NEXT LOGICAL STEP only, not the full path to completion

RULES:
- Tasks should be actionable work for THIS iteration
- DO NOT include tasks about signaling completion or emitting tags
- DO NOT include tasks that assume you need to "try again" or "revise" - the loop does that
- Build on previous progress if any
- Each task needs a 2-3 word summary describing the action (e.g. "drafting poem", "scoring result")"""

    messages = [{"role": "user", "content": "Generate tasks for this iteration."}]

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
        tools=[
            {
                "name": "create_tasks",
                "description": "Create tasks for this iteration",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {
                                        "type": "string",
                                        "description": "Description of the task",
                                    },
                                    "summary": {
                                        "type": "string",
                                        "description": "2-3 word action summary (e.g. 'drafting poem', 'scoring result')",
                                    },
                                },
                                "required": ["content", "summary"],
                            },
                        }
                    },
                    "required": ["tasks"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "create_tasks"},
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    return [{"content": t["content"], "summary": t["summary"], "status": "pending"} for t in tool_use.input["tasks"]]


@activity.defn
async def execute_task(input: ExecuteTaskInput) -> str:
    """Execute a single task."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    task_context = f"Task {input.task_index + 1}/{input.total_tasks}: {input.task_content}"

    system_msg = f"""You are completing ONE step in a multi-step workflow.

YOUR TASK (do ONLY this):
{input.task_content}

CRITICAL CONSTRAINTS:
- Complete ONLY the task above - nothing more
- Do NOT anticipate or work on future steps
- Do NOT look ahead at the overall goal
- If task says "write X", write X only - don't also revise/improve/evaluate
- Stay narrowly focused on the specific deliverable requested
- Stop when this single task is done

Build on any previous work shown in the conversation history.
Provide your output for this task only."""

    messages = list(input.history) + [{"role": "user", "content": task_context}]

    response = client.messages.create(
        model=input.model,
        max_tokens=4096,
        system=system_msg,
        messages=messages,
    )

    return response.content[0].text


@activity.defn
async def evaluate_iteration_completion(input: EvaluateIterationInput) -> EvaluateIterationOutput:
    """Evaluate if the overall task is complete after this iteration."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    outputs_text = "\n\n---\n\n".join(
        f"Output {i + 1}:\n{output}" for i, output in enumerate(input.task_outputs)
    )

    system = f"""Evaluate whether the original task has been completed based on this iteration's work.

Original task: {input.prompt}

{f"Previous progress: {input.progress_summary}" if input.progress_summary else ""}

This iteration's outputs:
{outputs_text}

Use the evaluate_iteration tool to provide your evaluation.

If the work is TRULY COMPLETE and satisfies the original task, include this exact tag in final_response:
<promise>{input.completion_promise}</promise>

CRITICAL RULES:
- Be STRICT: partial completion = NOT complete
- Only emit the promise tag if ALL requirements are satisfied
- Update progress summary with what was accomplished this iteration"""

    messages = [{"role": "user", "content": "Evaluate if the task is complete."}]

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,
        tools=[
            {
                "name": "evaluate_iteration",
                "description": "Evaluate iteration completion",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "completion_detected": {
                            "type": "boolean",
                            "description": "Whether the task is fully complete",
                        },
                        "progress_update": {
                            "type": "string",
                            "description": "What was accomplished this iteration",
                        },
                        "final_response": {
                            "type": "string",
                            "description": "Response text. Include <promise>PHRASE</promise> if complete.",
                        },
                    },
                    "required": ["completion_detected", "progress_update", "final_response"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "evaluate_iteration"},
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    result = tool_use.input

    # Build updated progress
    updated_progress = input.progress_summary
    if updated_progress:
        updated_progress += "\n"
    updated_progress += f"- Iteration: {result['progress_update']}"

    return EvaluateIterationOutput(
        updated_progress=updated_progress,
        completion_detected=result["completion_detected"],
        final_response=result["final_response"],
    )
