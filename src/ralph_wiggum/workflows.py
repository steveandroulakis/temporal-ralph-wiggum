"""Ralph Wiggum workflow implementation."""

from datetime import timedelta
from typing import Optional
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .models import (
        PRD,
        Story,
        RalphWorkflowInput,
        RalphWorkflowOutput,
        GeneratePRDInput,
        GenerateTasksInput,
        ExecuteTaskInput,
        EvaluateStoryInput,
        EvaluateOverallInput,
    )
    from .activities import (
        generate_prd,
        generate_tasks,
        execute_task,
        evaluate_story_completion,
        evaluate_overall_completion,
    )


@workflow.defn
class RalphWorkflow:
    """Temporal workflow implementing the PRD-driven Ralph Wiggum loop pattern."""

    def __init__(self) -> None:
        self._iteration = 0
        self._history: list = []
        self._prd: Optional[PRD] = None
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
    def get_prd(self) -> Optional[PRD]:
        """Query current PRD state."""
        return self._prd

    @workflow.query
    def get_progress_summary(self) -> str:
        """Query progress summary."""
        return self._progress_summary

    @workflow.query
    def get_current_story(self) -> Optional[Story]:
        """Query current in-progress story."""
        if self._prd:
            return self._prd.get_next_incomplete()
        return None

    @workflow.query
    def get_current_tasks(self) -> list:
        """Query current task list."""
        return self._current_tasks

    def _get_rolling_history(self) -> list:
        """Return last N messages for context."""
        return self._history[-self._history_window_size :]

    @workflow.run
    async def run(self, input: RalphWorkflowInput) -> RalphWorkflowOutput:
        """Run the PRD-driven Ralph Wiggum loop."""
        # Restore state from continue-as-new
        self._iteration = input.current_iteration
        self._history = list(input.conversation_history)
        self._prd = input.prd
        self._progress_summary = input.progress_summary
        self._history_window_size = input.history_window_size

        # Phase 1: Generate PRD (first run only)
        if self._prd is None:
            self._prd = await workflow.execute_activity(
                generate_prd,
                GeneratePRDInput(prompt=input.prompt, model=input.model),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=30),
                ),
                summary="generating prd from prompt",
            )

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
                        prd=self._prd,
                        progress_summary=self._progress_summary,
                    )
                )

            # Phase 2: Pick next incomplete story
            current_story = self._prd.get_next_incomplete()
            if current_story is None:
                break  # All stories complete, exit loop

            current_story.status = "in_progress"

            # Phase 3: Generate tasks for THIS story
            self._current_tasks = await workflow.execute_activity(
                generate_tasks,
                GenerateTasksInput(
                    story=current_story,
                    original_prompt=input.prompt,
                    progress_summary=self._progress_summary,
                    iteration=self._iteration,
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=30),
                ),
                summary=f"[{current_story.title[:25]}...] generating tasks",
            )

            # Phase 4: Execute all tasks, collect outputs
            task_outputs = []
            story_context = f"{current_story.title}: {current_story.description}"

            for i, task in enumerate(self._current_tasks):
                task["status"] = "in_progress"

                response = await workflow.execute_activity(
                    execute_task,
                    ExecuteTaskInput(
                        task_content=task["content"],
                        original_prompt=input.prompt,
                        story_context=story_context,
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
                    summary=f"[{current_story.title[:20]}...] task {i+1}/{len(self._current_tasks)}",
                )

                task["status"] = "completed"
                task_outputs.append(response)
                self._history.append({"role": "assistant", "content": response})

            # Phase 5: Evaluate story completion (END of iteration)
            eval_result = await workflow.execute_activity(
                evaluate_story_completion,
                EvaluateStoryInput(
                    story=current_story,
                    task_outputs=task_outputs,
                    original_prompt=input.prompt,
                    progress_summary=self._progress_summary,
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=30),
                ),
                summary=f"evaluating: {current_story.title[:30]}...?",
            )

            if eval_result.is_complete:
                current_story.status = "completed"
                current_story.completion_summary = eval_result.summary
            # else: stays in_progress, will be picked again next iteration

            self._progress_summary = eval_result.updated_progress
            self._iteration += 1

        # Phase 6: All stories complete OR max iterations hit
        if self._prd is not None and self._prd.all_complete():
            final_response = await workflow.execute_activity(
                evaluate_overall_completion,
                EvaluateOverallInput(
                    prd=self._prd,
                    original_prompt=input.prompt,
                    completion_promise=input.completion_promise,
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=30),
                ),
                summary="final validation - all stories",
            )

            # Check if promise tag is in the response
            promise_tag = f"<promise>{input.completion_promise}</promise>"
            completion_detected = promise_tag in final_response

            return RalphWorkflowOutput(
                completed=True,
                iterations_used=self._iteration,
                final_response=final_response,
                completion_detected=completion_detected,
            )

        # Max iterations reached without completion
        final_response = self._progress_summary if self._progress_summary else ""
        return RalphWorkflowOutput(
            completed=False,
            iterations_used=self._iteration,
            final_response=final_response,
            completion_detected=False,
        )
