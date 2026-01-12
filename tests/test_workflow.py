"""Tests for Ralph Wiggum workflow."""

import pytest
from unittest.mock import MagicMock, patch
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity

import sys
sys.path.insert(0, "/Users/steveandroulakis/Code/tmp/temporal-ralph-wiggum/src")

from ralph_wiggum.models import (
    RalphWorkflowInput,
    CallClaudeInput,
    CheckCompletionInput,
)
from ralph_wiggum.workflows import RalphWorkflow
from ralph_wiggum.activities import check_completion


class TestCheckCompletion:
    """Tests for check_completion activity."""

    @pytest.mark.asyncio
    async def test_detects_completion_promise(self):
        """Should detect when response contains completion promise."""
        input_data = CheckCompletionInput(
            response="Here is the result. <promise>COMPLETE</promise>",
            completion_promise="COMPLETE",
        )
        result = await check_completion(input_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_completion_promise(self):
        """Should return False when response doesn't contain promise."""
        input_data = CheckCompletionInput(
            response="Still working on the task...",
            completion_promise="COMPLETE",
        )
        result = await check_completion(input_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_partial_match_not_detected(self):
        """Should not detect partial matches of promise."""
        input_data = CheckCompletionInput(
            response="COMP is not COMPLETE",
            completion_promise="COMPLETE",
        )
        result = await check_completion(input_data)
        assert result is False


class TestCallClaude:
    """Tests for call_claude activity."""

    @pytest.mark.asyncio
    async def test_calls_anthropic_api(self):
        """Should call Anthropic API with correct parameters."""
        from ralph_wiggum.activities import call_claude

        input_data = CallClaudeInput(
            prompt="Write a haiku",
            history=[],
            model="claude-sonnet-4-5-20250514",
            iteration=0,
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="A haiku appears")]
            mock_client.messages.create.return_value = mock_response

            result = await call_claude(input_data)
            assert result == "A haiku appears"

    @pytest.mark.asyncio
    async def test_includes_iteration_context(self):
        """Should include iteration number in system message."""
        from ralph_wiggum.activities import call_claude

        input_data = CallClaudeInput(
            prompt="Write a haiku",
            history=[{"role": "assistant", "content": "Previous attempt"}],
            model="claude-sonnet-4-5-20250514",
            iteration=3,
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Another haiku")]
            mock_client.messages.create.return_value = mock_response

            await call_claude(input_data)

            # Verify system message includes iteration info
            call_args = mock_client.messages.create.call_args
            assert "iteration" in call_args.kwargs.get("system", "").lower()


class TestRalphWorkflow:
    """Integration tests for RalphWorkflow."""

    @pytest.mark.asyncio
    async def test_workflow_completes_on_promise(self):
        """Workflow should complete when Claude outputs the promise."""
        iteration_count = [0]

        @activity.defn(name="call_claude")
        async def mock_call_claude(input: CallClaudeInput) -> str:
            iteration_count[0] += 1
            if iteration_count[0] == 1:
                return "Working on it..."
            return "Done! <promise>COMPLETE</promise>"

        @activity.defn(name="check_completion")
        async def mock_check_completion(input: CheckCompletionInput) -> bool:
            return "<promise>" + input.completion_promise + "</promise>" in input.response

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[mock_call_claude, mock_check_completion],
            ):
                result = await env.client.execute_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-workflow-completes",
                    task_queue="ralph-test",
                )

                assert result.completed is True
                assert result.completion_detected is True
                assert result.iterations_used == 2

    @pytest.mark.asyncio
    async def test_workflow_stops_at_max_iterations(self):
        """Workflow should stop at max_iterations if promise never detected."""
        @activity.defn(name="call_claude")
        async def mock_call_claude(input: CallClaudeInput) -> str:
            return f"Still working... iteration {input.iteration}"

        @activity.defn(name="check_completion")
        async def mock_check_completion(input: CheckCompletionInput) -> bool:
            return False

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[mock_call_claude, mock_check_completion],
            ):
                result = await env.client.execute_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=3,
                    ),
                    id="test-workflow-max-iterations",
                    task_queue="ralph-test",
                )

                assert result.completed is False
                assert result.completion_detected is False
                assert result.iterations_used == 3

    @pytest.mark.asyncio
    async def test_workflow_preserves_history(self):
        """Workflow should pass conversation history to Claude activity."""
        captured_inputs = []

        @activity.defn(name="call_claude")
        async def mock_call_claude(input: CallClaudeInput) -> str:
            captured_inputs.append(input)
            if input.iteration >= 2:
                return "<promise>COMPLETE</promise>"
            return f"Response {input.iteration}"

        @activity.defn(name="check_completion")
        async def mock_check_completion(input: CheckCompletionInput) -> bool:
            return "<promise>COMPLETE</promise>" in input.response

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[mock_call_claude, mock_check_completion],
            ):
                await env.client.execute_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-workflow-history",
                    task_queue="ralph-test",
                )

            # Verify history grows with each iteration
            assert len(captured_inputs) == 3
            assert len(captured_inputs[0].history) == 0
            assert len(captured_inputs[1].history) == 1
            assert len(captured_inputs[2].history) == 2


class TestRalphWorkflowQueries:
    """Tests for RalphWorkflow query methods."""

    @pytest.mark.asyncio
    async def test_get_current_iteration_query(self):
        """Should return current iteration via query."""
        iteration_at_query = []

        @activity.defn(name="call_claude")
        async def mock_call_claude(input: CallClaudeInput) -> str:
            if input.iteration >= 2:
                return "<promise>COMPLETE</promise>"
            return f"Response {input.iteration}"

        @activity.defn(name="check_completion")
        async def mock_check_completion(input: CheckCompletionInput) -> bool:
            return "<promise>COMPLETE</promise>" in input.response

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[mock_call_claude, mock_check_completion],
            ):
                handle = await env.client.start_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-query-iteration",
                    task_queue="ralph-test",
                )

                result = await handle.result()

                # Query after completion - should be at final iteration
                iteration = await handle.query(RalphWorkflow.get_current_iteration)
                assert iteration == 3  # 0, 1, 2 + increment = 3

    @pytest.mark.asyncio
    async def test_get_history_query(self):
        """Should return conversation history via query."""
        @activity.defn(name="call_claude")
        async def mock_call_claude(input: CallClaudeInput) -> str:
            if input.iteration >= 1:
                return "<promise>COMPLETE</promise>"
            return "First response"

        @activity.defn(name="check_completion")
        async def mock_check_completion(input: CheckCompletionInput) -> bool:
            return "<promise>COMPLETE</promise>" in input.response

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[mock_call_claude, mock_check_completion],
            ):
                handle = await env.client.start_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-query-history",
                    task_queue="ralph-test",
                )

                result = await handle.result()

                # Query history after completion
                history = await handle.query(RalphWorkflow.get_history)
                assert len(history) == 2
                assert history[0]["role"] == "assistant"
                assert history[0]["content"] == "First response"
                assert history[1]["content"] == "<promise>COMPLETE</promise>"
