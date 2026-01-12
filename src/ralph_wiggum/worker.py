"""Worker for Ralph Wiggum workflow."""

import asyncio
from temporalio.client import Client
from temporalio.worker import Worker

from .workflows import RalphWorkflow
from .activities import check_completion, generate_plan, execute_task

TASK_QUEUE = "ralph-wiggum-queue"


async def run_worker():
    """Run the worker."""
    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RalphWorkflow],
        activities=[check_completion, generate_plan, execute_task],
    )

    print(f"Starting worker on task queue: {TASK_QUEUE}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
