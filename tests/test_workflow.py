"""Tests for Ralph Wiggum workflow."""

import pytest
from unittest.mock import MagicMock, patch
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity

import sys

sys.path.insert(0, "/Users/steveandroulakis/Code/tmp/temporal-ralph-wiggum/src")

from ralph_wiggum.models import (
    DEFAULT_MODEL,
    RalphWorkflowInput,
    DecideIterationInput,
    DecideIterationOutput,
    GenerateTasksInput,
    ExecuteTaskInput,
    EvaluateIterationInput,
    EvaluateIterationOutput,
)
from ralph_wiggum.workflows import RalphWorkflow


class TestDecideIterationMode:
    """Tests for decide_iteration_mode activity."""

    @pytest.mark.asyncio
    async def test_decides_single_mode(self):
        """Should decide single mode for simple tasks."""
        from ralph_wiggum.activities import decide_iteration_mode

        input_data = DecideIterationInput(
            prompt="Write a haiku",
            progress_summary="",
            history=[],
            iteration=0,
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.input = {
                "mode": "single",
                "single_task_content": "Write a haiku about coding",
                "rationale": "Simple creative task",
            }
            mock_response = MagicMock()
            mock_response.content = [mock_tool_use]
            mock_client.messages.create.return_value = mock_response

            result = await decide_iteration_mode(input_data)

            assert result.mode == "single"
            assert result.single_task_content == "Write a haiku about coding"

    @pytest.mark.asyncio
    async def test_decides_multi_mode(self):
        """Should decide multi mode for complex tasks."""
        from ralph_wiggum.activities import decide_iteration_mode

        input_data = DecideIterationInput(
            prompt="Build a calculator app",
            progress_summary="",
            history=[],
            iteration=0,
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.input = {
                "mode": "multi",
                "single_task_content": "",
                "rationale": "Multiple steps needed",
            }
            mock_response = MagicMock()
            mock_response.content = [mock_tool_use]
            mock_client.messages.create.return_value = mock_response

            result = await decide_iteration_mode(input_data)

            assert result.mode == "multi"


class TestGenerateTasks:
    """Tests for generate_tasks activity."""

    @pytest.mark.asyncio
    async def test_generates_tasks_for_iteration(self):
        """Should generate tasks for current iteration."""
        from ralph_wiggum.activities import generate_tasks

        input_data = GenerateTasksInput(
            prompt="Build a calculator",
            progress_summary="",
            history=[],
            iteration=0,
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.input = {
                "tasks": [
                    {"content": "Create add function", "summary": "creating function"},
                    {"content": "Write tests for add", "summary": "writing tests"},
                ]
            }
            mock_response = MagicMock()
            mock_response.content = [mock_tool_use]
            mock_client.messages.create.return_value = mock_response

            result = await generate_tasks(input_data)

            assert len(result) == 2
            assert result[0]["content"] == "Create add function"
            assert result[0]["status"] == "pending"


class TestExecuteTask:
    """Tests for execute_task activity."""

    @pytest.mark.asyncio
    async def test_calls_anthropic_api(self):
        """Should call Anthropic API with task context."""
        from ralph_wiggum.activities import execute_task

        input_data = ExecuteTaskInput(
            task_content="Create add function",
            task_summary="creating function",
            original_prompt="Build a calculator",
            history=[],
            model=DEFAULT_MODEL,
            iteration=0,
            task_index=0,
            total_tasks=2,
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="def add(a, b): return a + b")]
            mock_client.messages.create.return_value = mock_response

            result = await execute_task(input_data)

            assert result == "def add(a, b): return a + b"

            # Verify messages include task context
            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs.get("messages", [])
            assert any("Task 1/2" in m.get("content", "") for m in messages)


class TestEvaluateIterationCompletion:
    """Tests for evaluate_iteration_completion activity."""

    @pytest.mark.asyncio
    async def test_evaluates_iteration_complete(self):
        """Should evaluate if iteration is complete."""
        from ralph_wiggum.activities import evaluate_iteration_completion

        input_data = EvaluateIterationInput(
            prompt="Write a haiku",
            progress_summary="",
            task_outputs=["Haiku written successfully"],
            completion_promise="DONE",
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.input = {
                "completion_detected": True,
                "progress_update": "Haiku completed",
                "final_response": "Task complete! <promise>DONE</promise>",
            }
            mock_response = MagicMock()
            mock_response.content = [mock_tool_use]
            mock_client.messages.create.return_value = mock_response

            result = await evaluate_iteration_completion(input_data)

            assert result.completion_detected is True
            assert "Haiku completed" in result.updated_progress

    @pytest.mark.asyncio
    async def test_evaluates_iteration_incomplete(self):
        """Should detect incomplete iteration."""
        from ralph_wiggum.activities import evaluate_iteration_completion

        input_data = EvaluateIterationInput(
            prompt="Build a calculator",
            progress_summary="",
            task_outputs=["Started add function"],
            completion_promise="DONE",
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.input = {
                "completion_detected": False,
                "progress_update": "Started implementation",
                "final_response": "More work needed",
            }
            mock_response = MagicMock()
            mock_response.content = [mock_tool_use]
            mock_client.messages.create.return_value = mock_response

            result = await evaluate_iteration_completion(input_data)

            assert result.completion_detected is False


class TestRalphWorkflow:
    """Integration tests for RalphWorkflow."""

    @pytest.mark.asyncio
    async def test_workflow_completes_with_single_task(self):
        """Workflow should complete with single task mode."""

        @activity.defn(name="decide_iteration_mode")
        async def mock_decide(input: DecideIterationInput) -> DecideIterationOutput:
            return DecideIterationOutput(
                mode="single",
                single_task_content="Write the haiku",
                rationale="Simple task",
            )

        @activity.defn(name="generate_tasks")
        async def mock_generate_tasks(input: GenerateTasksInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return "Haiku written"

        @activity.defn(name="evaluate_iteration_completion")
        async def mock_evaluate(input: EvaluateIterationInput) -> EvaluateIterationOutput:
            return EvaluateIterationOutput(
                updated_progress="- Haiku completed",
                completion_detected=True,
                final_response=f"Done! <promise>{input.completion_promise}</promise>",
            )

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_decide,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate,
                ],
            ):
                result = await env.client.execute_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Write a haiku",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-workflow-single-task",
                    task_queue="ralph-test",
                )

                assert result.completed is True
                assert result.completion_detected is True
                assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_workflow_completes_with_multi_task(self):
        """Workflow should complete with multi task mode."""

        @activity.defn(name="decide_iteration_mode")
        async def mock_decide(input: DecideIterationInput) -> DecideIterationOutput:
            return DecideIterationOutput(
                mode="multi",
                single_task_content="",
                rationale="Multiple steps needed",
            )

        @activity.defn(name="generate_tasks")
        async def mock_generate_tasks(input: GenerateTasksInput) -> list:
            return [
                {"content": "Task 1", "status": "pending"},
                {"content": "Task 2", "status": "pending"},
            ]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return f"Output for task {input.task_index}"

        @activity.defn(name="evaluate_iteration_completion")
        async def mock_evaluate(input: EvaluateIterationInput) -> EvaluateIterationOutput:
            return EvaluateIterationOutput(
                updated_progress="- Tasks completed",
                completion_detected=True,
                final_response=f"Done! <promise>{input.completion_promise}</promise>",
            )

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_decide,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate,
                ],
            ):
                result = await env.client.execute_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Build something",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-workflow-multi-task",
                    task_queue="ralph-test",
                )

                assert result.completed is True
                assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_workflow_multiple_iterations(self):
        """Workflow should run multiple iterations until complete."""
        iteration_count = [0]

        @activity.defn(name="decide_iteration_mode")
        async def mock_decide(input: DecideIterationInput) -> DecideIterationOutput:
            return DecideIterationOutput(
                mode="single",
                single_task_content="Work on task",
                rationale="Focused work",
            )

        @activity.defn(name="generate_tasks")
        async def mock_generate_tasks(input: GenerateTasksInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return "Task output"

        @activity.defn(name="evaluate_iteration_completion")
        async def mock_evaluate(input: EvaluateIterationInput) -> EvaluateIterationOutput:
            iteration_count[0] += 1
            # Complete on 3rd iteration
            is_done = iteration_count[0] >= 3
            return EvaluateIterationOutput(
                updated_progress=f"- Iteration {iteration_count[0]}",
                completion_detected=is_done,
                final_response=f"<promise>{input.completion_promise}</promise>" if is_done else "More work needed",
            )

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_decide,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate,
                ],
            ):
                result = await env.client.execute_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-workflow-multiple-iterations",
                    task_queue="ralph-test",
                )

                assert result.completed is True
                assert result.iterations_used == 3

    @pytest.mark.asyncio
    async def test_workflow_stops_at_max_iterations(self):
        """Workflow should stop at max_iterations if never complete."""

        @activity.defn(name="decide_iteration_mode")
        async def mock_decide(input: DecideIterationInput) -> DecideIterationOutput:
            return DecideIterationOutput(
                mode="single",
                single_task_content="Work on task",
                rationale="Focused work",
            )

        @activity.defn(name="generate_tasks")
        async def mock_generate_tasks(input: GenerateTasksInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return "Task output"

        @activity.defn(name="evaluate_iteration_completion")
        async def mock_evaluate(input: EvaluateIterationInput) -> EvaluateIterationOutput:
            # Never complete
            return EvaluateIterationOutput(
                updated_progress="- Still working",
                completion_detected=False,
                final_response="More work needed",
            )

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_decide,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate,
                ],
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


class TestRalphWorkflowQueries:
    """Tests for RalphWorkflow query methods."""

    @pytest.mark.asyncio
    async def test_get_progress_summary_query(self):
        """Should return progress summary via query."""

        @activity.defn(name="decide_iteration_mode")
        async def mock_decide(input: DecideIterationInput) -> DecideIterationOutput:
            return DecideIterationOutput(
                mode="single",
                single_task_content="Do work",
                rationale="Simple",
            )

        @activity.defn(name="generate_tasks")
        async def mock_generate_tasks(input: GenerateTasksInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return "Output"

        @activity.defn(name="evaluate_iteration_completion")
        async def mock_evaluate(input: EvaluateIterationInput) -> EvaluateIterationOutput:
            return EvaluateIterationOutput(
                updated_progress="- Task completed successfully",
                completion_detected=True,
                final_response=f"<promise>{input.completion_promise}</promise>",
            )

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_decide,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate,
                ],
            ):
                handle = await env.client.start_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-query-progress",
                    task_queue="ralph-test",
                )

                await handle.result()

                progress = await handle.query(RalphWorkflow.get_progress_summary)
                assert "Task completed successfully" in progress
