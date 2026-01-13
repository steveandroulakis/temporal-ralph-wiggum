"""Ralph Wiggum workflow implementation."""

from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .models import (
        RalphWorkflowInput,
        RalphWorkflowOutput,
        DecideIterationInput,
        GenerateTasksInput,
        ExecuteTaskInput,
        EvaluateIterationInput,
    )
    from .activities import (
        decide_iteration_mode,
        generate_tasks,
        execute_task,
        evaluate_iteration_completion,
    )


@workflow.defn
class RalphWorkflow:
    """Temporal workflow implementing the iteration-decision Ralph Wiggum loop."""

    def __init__(self) -> None:
        self._iteration = 0
        self._history: list = []
        self._progress_summary: str = ""
        self._current_tasks: list = []
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
    def get_progress_summary(self) -> str:
        """Query progress summary."""
        return self._progress_summary

    @workflow.query
    def get_current_tasks(self) -> list:
        """Query current task list."""
        return self._current_tasks

    def _get_rolling_history(self) -> list:
        """Return last N messages for context."""
        return self._history[-self._history_window_size :]

    @workflow.run
    async def run(self, input: RalphWorkflowInput) -> RalphWorkflowOutput:
        """Run the iteration-decision Ralph Wiggum loop."""
        # Restore state from continue-as-new
        self._iteration = input.current_iteration
        self._history = list(input.conversation_history)
        self._progress_summary = input.progress_summary
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
                        progress_summary=self._progress_summary,
                    )
                )

            # Step 1: Decide iteration mode (single vs multi)
            decision = await workflow.execute_activity(
                decide_iteration_mode,
                DecideIterationInput(
                    prompt=input.prompt,
                    progress_summary=self._progress_summary,
                    history=self._get_rolling_history(),
                    iteration=self._iteration,
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=30),
                ),
                summary=f"iteration {self._iteration + 1}: deciding mode",
            )

            task_outputs = []

            if decision.mode == "single":
                # Single task mode - execute one task directly
                response = await workflow.execute_activity(
                    execute_task,
                    ExecuteTaskInput(
                        task_content=decision.single_task_content,
                        task_summary="single task",
                        original_prompt=input.prompt,
                        history=self._get_rolling_history(),
                        model=input.model,
                        iteration=self._iteration,
                        task_index=0,
                        total_tasks=1,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=RetryPolicy(
                        maximum_attempts=3,
                        initial_interval=timedelta(seconds=1),
                        maximum_interval=timedelta(seconds=30),
                    ),
                    summary=f"iteration {self._iteration + 1}: single task",
                )
                task_outputs.append(response)
                self._history.append({"role": "assistant", "content": response})

            else:
                # Multi-task mode - generate tasks then execute each
                self._current_tasks = await workflow.execute_activity(
                    generate_tasks,
                    GenerateTasksInput(
                        prompt=input.prompt,
                        progress_summary=self._progress_summary,
                        history=self._get_rolling_history(),
                        iteration=self._iteration,
                    ),
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=RetryPolicy(
                        maximum_attempts=3,
                        initial_interval=timedelta(seconds=1),
                        maximum_interval=timedelta(seconds=30),
                    ),
                    summary=f"iteration {self._iteration + 1}: generating tasks",
                )

                for i, task in enumerate(self._current_tasks):
                    task["status"] = "in_progress"
                    task_summary = task.get("summary", "task")

                    response = await workflow.execute_activity(
                        execute_task,
                        ExecuteTaskInput(
                            task_content=task["content"],
                            task_summary=task_summary,
                            original_prompt=input.prompt,
                            history=self._get_rolling_history(),
                            model=input.model,
                            iteration=self._iteration,
                            task_index=i,
                            total_tasks=len(self._current_tasks),
                        ),
                        start_to_close_timeout=timedelta(minutes=5),
                        retry_policy=RetryPolicy(
                            maximum_attempts=3,
                            initial_interval=timedelta(seconds=1),
                            maximum_interval=timedelta(seconds=30),
                        ),
                        summary=f"iteration {self._iteration + 1}: task {i + 1}/{len(self._current_tasks)} - {task_summary}",
                    )

                    task["status"] = "completed"
                    task_outputs.append(response)
                    self._history.append({"role": "assistant", "content": response})

            # Step 3: Evaluate iteration completion
            eval_result = await workflow.execute_activity(
                evaluate_iteration_completion,
                EvaluateIterationInput(
                    prompt=input.prompt,
                    progress_summary=self._progress_summary,
                    task_outputs=task_outputs,
                    completion_promise=input.completion_promise,
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=30),
                ),
                summary=f"iteration {self._iteration + 1}: evaluating completion",
            )

            self._progress_summary = eval_result.updated_progress

            if eval_result.completion_detected:
                # Check for promise tag in final_response
                promise_tag = f"<promise>{input.completion_promise}</promise>"
                has_promise = promise_tag in eval_result.final_response

                return RalphWorkflowOutput(
                    completed=True,
                    iterations_used=self._iteration + 1,
                    final_response=eval_result.final_response,
                    completion_detected=has_promise,
                )

            self._iteration += 1

        # Max iterations reached without completion
        return RalphWorkflowOutput(
            completed=False,
            iterations_used=self._iteration,
            final_response=self._progress_summary,
            completion_detected=False,
        )
