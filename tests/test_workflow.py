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
    CheckCompletionInput,
    GeneratePlanInput,
    ExecuteTaskInput,
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


class TestGeneratePlan:
    """Tests for generate_plan activity."""

    @pytest.mark.asyncio
    async def test_calls_anthropic_api_with_tool(self):
        """Should call Anthropic API with create_plan tool."""
        from ralph_wiggum.activities import generate_plan

        input_data = GeneratePlanInput(
            prompt="1. Write a poem. 2. Translate to French.",
            iteration=0,
            history=[],
            previous_plan=None,
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.input = {"tasks": [{"content": "Write a poem"}, {"content": "Translate to French"}]}
            mock_response = MagicMock()
            mock_response.content = [mock_tool_use]
            mock_client.messages.create.return_value = mock_response

            result = await generate_plan(input_data)
            assert len(result) == 2
            assert result[0]["content"] == "Write a poem"
            assert result[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_includes_previous_plan_in_system(self):
        """Should include previous plan in system message when provided."""
        from ralph_wiggum.activities import generate_plan

        input_data = GeneratePlanInput(
            prompt="Write a poem",
            iteration=1,
            history=[],
            previous_plan=[{"content": "Old task", "status": "completed"}],
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.input = {"tasks": [{"content": "New task"}]}
            mock_response = MagicMock()
            mock_response.content = [mock_tool_use]
            mock_client.messages.create.return_value = mock_response

            await generate_plan(input_data)

            # Verify system message includes previous plan
            call_args = mock_client.messages.create.call_args
            assert "Old task" in call_args.kwargs.get("system", "")


class TestExecuteTask:
    """Tests for execute_task activity."""

    @pytest.mark.asyncio
    async def test_calls_anthropic_api(self):
        """Should call Anthropic API with task context."""
        from ralph_wiggum.activities import execute_task

        input_data = ExecuteTaskInput(
            task_content="Write a haiku",
            original_prompt="Write poems",
            history=[],
            model="claude-sonnet-4-5-20250514",
            iteration=0,
            task_index=0,
            total_tasks=2,
            completion_promise="COMPLETE",
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="A haiku appears")]
            mock_client.messages.create.return_value = mock_response

            result = await execute_task(input_data)
            assert result == "A haiku appears"

            # Verify system message includes task context
            call_args = mock_client.messages.create.call_args
            assert "Task 1/2" in call_args.kwargs.get("system", "")


class TestRalphWorkflow:
    """Integration tests for RalphWorkflow."""

    @pytest.mark.asyncio
    async def test_workflow_completes_on_promise(self):
        """Workflow should complete when a task outputs the promise."""
        task_count = [0]

        @activity.defn(name="generate_plan")
        async def mock_generate_plan(input: GeneratePlanInput) -> list:
            return [{"content": "Task 1", "status": "pending"}, {"content": "Task 2", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            task_count[0] += 1
            if task_count[0] == 2:
                return "Done! <promise>COMPLETE</promise>"
            return "Working on it..."

        @activity.defn(name="check_completion")
        async def mock_check_completion(input: CheckCompletionInput) -> bool:
            return "<promise>" + input.completion_promise + "</promise>" in input.response

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[mock_generate_plan, mock_execute_task, mock_check_completion],
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
                # 1 iteration, completed on 2nd task
                assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_workflow_stops_at_max_iterations(self):
        """Workflow should stop at max_iterations if promise never detected."""
        @activity.defn(name="generate_plan")
        async def mock_generate_plan(input: GeneratePlanInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return f"Still working... iteration {input.iteration}"

        @activity.defn(name="check_completion")
        async def mock_check_completion(input: CheckCompletionInput) -> bool:
            return False

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[mock_generate_plan, mock_execute_task, mock_check_completion],
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
        """Workflow should pass conversation history to execute_task activity."""
        captured_inputs = []

        @activity.defn(name="generate_plan")
        async def mock_generate_plan(input: GeneratePlanInput) -> list:
            return [{"content": "Task 1", "status": "pending"}, {"content": "Task 2", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            captured_inputs.append(input)
            if input.task_index == 1 and input.iteration >= 1:
                return "<promise>COMPLETE</promise>"
            return f"Response iter={input.iteration} task={input.task_index}"

        @activity.defn(name="check_completion")
        async def mock_check_completion(input: CheckCompletionInput) -> bool:
            return "<promise>COMPLETE</promise>" in input.response

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[mock_generate_plan, mock_execute_task, mock_check_completion],
            ):
                await env.client.execute_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                        history_window_size=3,
                    ),
                    id="test-workflow-history",
                    task_queue="ralph-test",
                )

            # First iteration: 2 tasks, second iteration: 2 tasks = 4 total calls
            assert len(captured_inputs) == 4
            # First task of first iteration: no history
            assert len(captured_inputs[0].history) == 0
            # Second task of first iteration: 1 history item
            assert len(captured_inputs[1].history) == 1
            # First task of second iteration: 2 history items
            assert len(captured_inputs[2].history) == 2


class TestRalphWorkflowQueries:
    """Tests for RalphWorkflow query methods."""

    @pytest.mark.asyncio
    async def test_get_current_iteration_query(self):
        """Should return current iteration via query."""
        @activity.defn(name="generate_plan")
        async def mock_generate_plan(input: GeneratePlanInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
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
                activities=[mock_generate_plan, mock_execute_task, mock_check_completion],
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

                # Query after completion - should be at final iteration (3 because it returned on iteration 2 + 1)
                iteration = await handle.query(RalphWorkflow.get_current_iteration)
                assert iteration == 2  # 0, 1, 2 - completed on iteration 2

    @pytest.mark.asyncio
    async def test_get_todos_query(self):
        """Should return current todos via query."""
        @activity.defn(name="generate_plan")
        async def mock_generate_plan(input: GeneratePlanInput) -> list:
            return [{"content": "Task A", "status": "pending"}, {"content": "Task B", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            if input.task_index == 1:
                return "<promise>COMPLETE</promise>"
            return "Response"

        @activity.defn(name="check_completion")
        async def mock_check_completion(input: CheckCompletionInput) -> bool:
            return "<promise>COMPLETE</promise>" in input.response

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[mock_generate_plan, mock_execute_task, mock_check_completion],
            ):
                handle = await env.client.start_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-query-todos",
                    task_queue="ralph-test",
                )

                result = await handle.result()

                # Query todos after completion
                todos = await handle.query(RalphWorkflow.get_todos)
                assert len(todos) == 2
                assert todos[0]["content"] == "Task A"
                assert todos[0]["status"] == "completed"
                assert todos[1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_history_query(self):
        """Should return conversation history via query."""
        @activity.defn(name="generate_plan")
        async def mock_generate_plan(input: GeneratePlanInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
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
                activities=[mock_generate_plan, mock_execute_task, mock_check_completion],
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
