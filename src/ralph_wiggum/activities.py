"""Activities for Ralph Wiggum workflow."""

import os
import anthropic
from temporalio import activity

from .models import (
    DEFAULT_MODEL,
    PRD,
    Story,
    GeneratePRDInput,
    GenerateTasksInput,
    ExecuteTaskInput,
    EvaluateStoryInput,
    EvaluateStoryOutput,
    EvaluateOverallInput,
)


@activity.defn
async def generate_prd(input: GeneratePRDInput) -> PRD:
    """Generate high-level stories from prompt. Run once on first iteration."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system = """Break down the user's task into 3-7 high-level stories.
Each story should be an independently completable deliverable.
Use the create_prd tool to output your stories.

CRITICAL RULES:
- Each story = a distinct piece of work that can be verified when complete
- Stories should be ordered by logical dependency/priority
- DO NOT include meta-tasks like "signal completion", "emit done tag", or "finalize"
- DO NOT include setup/teardown stories unless explicitly part of the task
- Focus on actual deliverables the user asked for"""

    messages = [{"role": "user", "content": input.prompt}]

    response = client.messages.create(
        model=input.model,
        max_tokens=2048,
        system=system,
        messages=messages,
        tools=[
            {
                "name": "create_prd",
                "description": "Create a product requirements document with stories",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "stories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "description": "Short title for the story",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Detailed description of what needs to be done",
                                    },
                                },
                                "required": ["title", "description"],
                            },
                        }
                    },
                    "required": ["stories"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "create_prd"},
    )

    # Extract stories from tool call
    tool_use = next(b for b in response.content if b.type == "tool_use")
    stories = [
        Story(
            id=f"story-{i + 1}",
            title=s["title"],
            description=s["description"],
            status="pending",
        )
        for i, s in enumerate(tool_use.input["stories"])
    ]

    return PRD(stories=stories)


@activity.defn
async def generate_tasks(input: GenerateTasksInput) -> list[dict]:
    """Generate tasks for a SINGLE story. Called each iteration."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system = f"""Generate 2-5 concrete tasks to complete ONLY this story.

Story: {input.story.title}
Description: {input.story.description}

{f"Progress so far: {input.progress_summary}" if input.progress_summary else ""}

CRITICAL RULES:
- ONLY generate tasks for the story above - ignore any broader context
- The story description is your SOLE source of requirements
- DO NOT include tasks from other stories or future work
- DO NOT include tasks about signaling completion or emitting tags
- Each task should directly contribute to completing THIS story's description
- Iteration: {input.iteration}"""

    messages = [
        {
            "role": "user",
            "content": f"Create tasks for story: {input.story.title}\n\n{input.story.description}",
        }
    ]

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
        tools=[
            {
                "name": "create_tasks",
                "description": "Create tasks for the current story",
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
                                },
                                "required": ["content"],
                            },
                        }
                    },
                    "required": ["tasks"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "create_tasks"},
    )

    # Extract tasks from tool call
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return [{"content": t["content"], "status": "pending"} for t in tool_use.input["tasks"]]


@activity.defn
async def execute_task(input: ExecuteTaskInput) -> str:
    """Execute a single task from the plan."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    task_context = f"Task {input.task_index + 1}/{input.total_tasks}: {input.task_content}"

    system_msg = f"""You are working on a multi-step task. This is iteration {input.iteration + 1}.

Original task: {input.original_prompt}

Current story: {input.story_context}

Current step: {task_context}

Focus on completing THIS specific step thoroughly.
Build on any previous work shown in the conversation history.
Provide concrete output, code, or results as appropriate."""

    messages = list(input.history) + [{"role": "user", "content": task_context}]

    response = client.messages.create(
        model=input.model,
        max_tokens=4096,
        system=system_msg,
        messages=messages,
    )

    return response.content[0].text


@activity.defn
async def evaluate_story_completion(input: EvaluateStoryInput) -> EvaluateStoryOutput:
    """Evaluate if a story is complete based on task outputs."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Format task outputs for evaluation
    outputs_text = "\n\n---\n\n".join(
        f"Task output {i + 1}:\n{output}" for i, output in enumerate(input.task_outputs)
    )

    system = f"""Evaluate whether a story has been completed based on the task outputs.

Story: {input.story.title}
Description: {input.story.description}

Original task context: {input.original_prompt}

{f"Previous progress: {input.progress_summary}" if input.progress_summary else ""}

Use the evaluate_story tool to provide your evaluation.

EVALUATION CRITERIA:
- Be STRICT: partial completion = NOT complete
- The story's requirements must be fully satisfied
- If outputs only partially address the story, mark as incomplete
- Consider: Did the outputs actually deliver what the story requires?"""

    messages = [
        {
            "role": "user",
            "content": f"Evaluate if this story is complete based on these outputs:\n\n{outputs_text}",
        }
    ]

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages,
        tools=[
            {
                "name": "evaluate_story",
                "description": "Evaluate story completion",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "is_complete": {
                            "type": "boolean",
                            "description": "Whether the story is fully complete",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of what was accomplished",
                        },
                        "progress_update": {
                            "type": "string",
                            "description": "Update to add to progress summary",
                        },
                    },
                    "required": ["is_complete", "summary", "progress_update"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "evaluate_story"},
    )

    # Extract evaluation from tool call
    tool_use = next(b for b in response.content if b.type == "tool_use")
    result = tool_use.input

    # Build updated progress summary
    updated_progress = input.progress_summary
    if updated_progress:
        updated_progress += "\n"
    updated_progress += f"- {input.story.title}: {result['progress_update']}"

    return EvaluateStoryOutput(
        is_complete=result["is_complete"],
        summary=result["summary"],
        updated_progress=updated_progress,
    )


@activity.defn
async def evaluate_overall_completion(input: EvaluateOverallInput) -> str:
    """Final check when all stories complete. Returns response with promise tag if truly done."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Format PRD status
    prd_summary = "\n".join(
        f"- [{s.status.upper()}] {s.title}: {s.completion_summary or s.description}"
        for s in input.prd.stories
    )

    system = f"""All stories have been marked complete. Review the overall work and confirm completion.

Original task: {input.original_prompt}

PRD Status:
{prd_summary}

If you are SATISFIED that all work is truly complete, output the completion signal.
If something is missing or incomplete, explain what needs more work.

To signal completion, output EXACTLY this tag:
<promise>{input.completion_promise}</promise>

CRITICAL: Only output the promise tag if the work is TRULY complete."""

    messages = [
        {
            "role": "user",
            "content": "Review the completed work and determine if the original task is satisfied.",
        }
    ]

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,
    )

    return response.content[0].text
