#!/usr/bin/env python3
"""CLI to run Ralph Wiggum workflow."""

import argparse
import asyncio
import uuid
from temporalio.client import Client

import sys
sys.path.insert(0, "src")

from ralph_wiggum.workflows import RalphWorkflow
from ralph_wiggum.models import RalphWorkflowInput

TASK_QUEUE = "ralph-wiggum-queue"


async def run_workflow(args):
    """Execute the Ralph Wiggum workflow."""
    client = await Client.connect("localhost:7233")

    workflow_id = f"ralph-loop-{uuid.uuid4().hex[:8]}"

    print(f"Starting workflow: {workflow_id}")
    print(f"Prompt: {args.prompt}")
    print(f"Completion promise: {args.completion_promise}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Model: {args.model}")
    print("-" * 50)

    result = await client.execute_workflow(
        RalphWorkflow.run,
        RalphWorkflowInput(
            prompt=args.prompt,
            completion_promise=args.completion_promise,
            max_iterations=args.max_iterations,
            model=args.model,
        ),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    print("-" * 50)
    print(f"Workflow completed!")
    print(f"  Completed: {result.completed}")
    print(f"  Iterations used: {result.iterations_used}")
    print(f"  Completion detected: {result.completion_detected}")
    print("-" * 50)
    print("Final response:")
    print(result.final_response)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run Ralph Wiggum workflow")
    parser.add_argument(
        "--prompt",
        "-p",
        required=True,
        help="The task prompt for Claude",
    )
    parser.add_argument(
        "--completion-promise",
        "-c",
        default="COMPLETE",
        help="Phrase that signals completion (default: COMPLETE)",
    )
    parser.add_argument(
        "--max-iterations",
        "-m",
        type=int,
        default=10,
        help="Maximum iterations (default: 10)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250514",
        help="Claude model to use (default: claude-sonnet-4-5-20250514)",
    )

    args = parser.parse_args()
    asyncio.run(run_workflow(args))


if __name__ == "__main__":
    main()
