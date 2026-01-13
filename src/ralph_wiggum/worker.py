"""Worker for Ralph Wiggum workflow."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from temporalio.client import Client
from temporalio.worker import Worker

from .workflows import RalphWorkflow
from .activities import (
    generate_prd,
    generate_tasks,
    execute_task,
    evaluate_story_completion,
    evaluate_overall_completion,
)

TASK_QUEUE = "ralph-wiggum-queue"


async def run_worker():
    """Run the worker."""
    client = await Client.connect("localhost:7233")

    # Use ThreadPoolExecutor so activities don't block the event loop
    # This allows queries to respond while activities are running
    with ThreadPoolExecutor(max_workers=100) as activity_executor:
        worker = Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[RalphWorkflow],
            activities=[
                generate_prd,
                generate_tasks,
                execute_task,
                evaluate_story_completion,
                evaluate_overall_completion,
            ],
            activity_executor=activity_executor,
        )

        print(f"Starting worker on task queue: {TASK_QUEUE}")
        await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
