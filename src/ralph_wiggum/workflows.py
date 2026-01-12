"""Ralph Wiggum workflow implementation."""

from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .models import (
        RalphWorkflowInput,
        RalphWorkflowOutput,
        CallClaudeInput,
        CheckCompletionInput,
    )
    from .activities import call_claude, check_completion


@workflow.defn
class RalphWorkflow:
    """Temporal workflow implementing the Ralph Wiggum loop pattern."""

    def __init__(self) -> None:
        self._iteration = 0
        self._history: list = []

    @workflow.query
    def get_current_iteration(self) -> int:
        """Query current iteration number."""
        return self._iteration

    @workflow.query
    def get_history(self) -> list:
        """Query conversation history."""
        return self._history

    @workflow.run
    async def run(self, input: RalphWorkflowInput) -> RalphWorkflowOutput:
        """Run the Ralph Wiggum loop."""
        self._iteration = input.current_iteration
        self._history = list(input.conversation_history)  # Copy to avoid mutation

        while self._iteration < input.max_iterations:
            # Check if we should continue-as-new (event history too large)
            if workflow.info().is_continue_as_new_suggested():
                workflow.continue_as_new(
                    RalphWorkflowInput(
                        prompt=input.prompt,
                        completion_promise=input.completion_promise,
                        max_iterations=input.max_iterations,
                        model=input.model,
                        current_iteration=self._iteration,
                        conversation_history=self._history,
                    )
                )

            # Call Claude
            response = await workflow.execute_activity(
                call_claude,
                CallClaudeInput(
                    prompt=input.prompt,
                    history=self._history,
                    model=input.model,
                    iteration=self._iteration,
                ),
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(
                    maximum_attempts=3,
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=30),
                ),
            )

            # Update history with assistant response
            self._history.append({"role": "assistant", "content": response})
            self._iteration += 1

            # Check completion
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
                    iterations_used=self._iteration,
                    final_response=response,
                    completion_detected=True,
                )

        # Max iterations reached
        final_response = self._history[-1]["content"] if self._history else ""
        return RalphWorkflowOutput(
            completed=False,
            iterations_used=self._iteration,
            final_response=final_response,
            completion_detected=False,
        )
