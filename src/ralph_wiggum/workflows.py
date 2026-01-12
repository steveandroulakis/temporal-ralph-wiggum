"""Ralph Wiggum workflow implementation."""

from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .models import (
        RalphWorkflowInput,
        RalphWorkflowOutput,
        CheckCompletionInput,
        GeneratePlanInput,
        ExecuteTaskInput,
    )
    from .activities import check_completion, generate_plan, execute_task


@workflow.defn
class RalphWorkflow:
    """Temporal workflow implementing the Ralph Wiggum loop pattern with dynamic planning."""

    def __init__(self) -> None:
        self._iteration = 0
        self._history: list = []
        self._todos: list = []
        self._history_window_size = 3

    @workflow.query
    def get_current_iteration(self) -> int:
        """Query current iteration number."""
        return self._iteration

    @workflow.query
    def get_history(self) -> list:
        """Query conversation history."""
        return self._history

    @workflow.query
    def get_todos(self) -> list:
        """Query current todo list."""
        return self._todos

    def _get_rolling_history(self) -> list:
        """Return last N messages for context."""
        return self._history[-self._history_window_size:]

    @workflow.run
    async def run(self, input: RalphWorkflowInput) -> RalphWorkflowOutput:
        """Run the Ralph Wiggum loop with dynamic planning."""
        self._iteration = input.current_iteration
        self._history = list(input.conversation_history)
        self._todos = list(input.previous_plan)
        self._history_window_size = input.history_window_size

        while self._iteration < input.max_iterations:
            # Check if we should continue-as-new (event history too large)
            if workflow.info().is_continue_as_new_suggested():
                workflow.continue_as_new(
                    RalphWorkflowInput(
                        prompt=input.prompt,
                        completion_promise=input.completion_promise,
                        max_iterations=input.max_iterations,
                        model=input.model,
                        history_window_size=input.history_window_size,
                        current_iteration=self._iteration,
                        conversation_history=self._history,
                        previous_plan=self._todos,
                    )
                )

            # 1. Generate plan (with rolling history + previous plan for context)
            previous_plan = self._todos if self._iteration > 0 else None
            self._todos = await workflow.execute_activity(
                generate_plan,
                GeneratePlanInput(
                    prompt=input.prompt,
                    iteration=self._iteration,
                    history=self._get_rolling_history(),
                    previous_plan=previous_plan,
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=30),
                ),
            )

            # 2. Execute each task
            for i, task in enumerate(self._todos):
                task["status"] = "in_progress"

                response = await workflow.execute_activity(
                    execute_task,
                    ExecuteTaskInput(
                        task_content=task["content"],
                        original_prompt=input.prompt,
                        history=self._get_rolling_history(),
                        model=input.model,
                        iteration=self._iteration,
                        task_index=i,
                        total_tasks=len(self._todos),
                        completion_promise=input.completion_promise,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=RetryPolicy(
                        maximum_attempts=3,
                        initial_interval=timedelta(seconds=1),
                        maximum_interval=timedelta(seconds=30),
                    ),
                )

                task["status"] = "completed"
                self._history.append({"role": "assistant", "content": response})

                # 3. Check completion after each task
                is_complete = await workflow.execute_activity(
                    check_completion,
                    CheckCompletionInput(
                        response=response,
                        completion_promise=input.completion_promise,
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                )

                if is_complete:
                    return RalphWorkflowOutput(
                        completed=True,
                        iterations_used=self._iteration + 1,
                        final_response=response,
                        completion_detected=True,
                    )

            self._iteration += 1

        # Max iterations reached
        final_response = self._history[-1]["content"] if self._history else ""
        return RalphWorkflowOutput(
            completed=False,
            iterations_used=self._iteration,
            final_response=final_response,
            completion_detected=False,
        )
