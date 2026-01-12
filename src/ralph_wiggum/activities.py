"""Activities for Ralph Wiggum workflow."""

import os
import anthropic
from temporalio import activity

from .models import CallClaudeInput, CheckCompletionInput


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
