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
    PRD,
    Story,
    RalphWorkflowInput,
    GeneratePRDInput,
    GenerateTasksInput,
    ExecuteTaskInput,
    EvaluateStoryInput,
    EvaluateStoryOutput,
    EvaluateOverallInput,
)
from ralph_wiggum.workflows import RalphWorkflow


class TestPRDModel:
    """Tests for PRD and Story models."""

    def test_get_next_incomplete_returns_first_pending(self):
        """Should return first non-completed story."""
        prd = PRD(
            stories=[
                Story(id="1", title="Story 1", description="Desc 1", status="completed"),
                Story(id="2", title="Story 2", description="Desc 2", status="pending"),
                Story(id="3", title="Story 3", description="Desc 3", status="pending"),
            ]
        )
        story = prd.get_next_incomplete()
        assert story is not None
        assert story.id == "2"

    def test_get_next_incomplete_returns_in_progress(self):
        """Should return in_progress story."""
        prd = PRD(
            stories=[
                Story(id="1", title="Story 1", description="Desc 1", status="completed"),
                Story(
                    id="2", title="Story 2", description="Desc 2", status="in_progress"
                ),
            ]
        )
        story = prd.get_next_incomplete()
        assert story is not None
        assert story.id == "2"

    def test_get_next_incomplete_returns_none_when_all_complete(self):
        """Should return None when all stories are completed."""
        prd = PRD(
            stories=[
                Story(id="1", title="Story 1", description="Desc 1", status="completed"),
                Story(id="2", title="Story 2", description="Desc 2", status="completed"),
            ]
        )
        story = prd.get_next_incomplete()
        assert story is None

    def test_all_complete_true(self):
        """Should return True when all stories completed."""
        prd = PRD(
            stories=[
                Story(id="1", title="Story 1", description="Desc 1", status="completed"),
                Story(id="2", title="Story 2", description="Desc 2", status="completed"),
            ]
        )
        assert prd.all_complete() is True

    def test_all_complete_false(self):
        """Should return False when some stories pending."""
        prd = PRD(
            stories=[
                Story(id="1", title="Story 1", description="Desc 1", status="completed"),
                Story(id="2", title="Story 2", description="Desc 2", status="pending"),
            ]
        )
        assert prd.all_complete() is False

    def test_all_complete_empty_prd(self):
        """Should return False for empty PRD."""
        prd = PRD(stories=[])
        assert prd.all_complete() is False


class TestGeneratePRD:
    """Tests for generate_prd activity."""

    @pytest.mark.asyncio
    async def test_calls_anthropic_api_with_tool(self):
        """Should call Anthropic API with create_prd tool."""
        from ralph_wiggum.activities import generate_prd

        input_data = GeneratePRDInput(
            prompt="Build a calculator app",
            model=DEFAULT_MODEL,
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.input = {
                "stories": [
                    {"title": "Add operation", "description": "Implement addition"},
                    {"title": "Subtract operation", "description": "Implement subtraction"},
                ]
            }
            mock_response = MagicMock()
            mock_response.content = [mock_tool_use]
            mock_client.messages.create.return_value = mock_response

            result = await generate_prd(input_data)

            assert len(result.stories) == 2
            assert result.stories[0].title == "Add operation"
            assert result.stories[0].id == "story-1"
            assert result.stories[0].status == "pending"


class TestGenerateTasks:
    """Tests for generate_tasks activity."""

    @pytest.mark.asyncio
    async def test_generates_tasks_for_story(self):
        """Should generate tasks for a single story."""
        from ralph_wiggum.activities import generate_tasks

        story = Story(
            id="story-1",
            title="Add operation",
            description="Implement addition functionality",
        )
        input_data = GenerateTasksInput(
            story=story,
            original_prompt="Build a calculator",
            progress_summary="",
            iteration=0,
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.input = {
                "tasks": [
                    {"content": "Create add function"},
                    {"content": "Write tests for add"},
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
            original_prompt="Build a calculator",
            story_context="Add operation: Implement addition functionality",
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

            # Verify system message includes story context
            call_args = mock_client.messages.create.call_args
            system = call_args.kwargs.get("system", "")
            assert "Add operation" in system
            assert "Task 1/2" in system


class TestEvaluateStoryCompletion:
    """Tests for evaluate_story_completion activity."""

    @pytest.mark.asyncio
    async def test_evaluates_story_completion(self):
        """Should evaluate if story is complete."""
        from ralph_wiggum.activities import evaluate_story_completion

        story = Story(
            id="story-1",
            title="Add operation",
            description="Implement addition",
        )
        input_data = EvaluateStoryInput(
            story=story,
            task_outputs=["def add(a, b): return a + b", "Tests pass"],
            original_prompt="Build calculator",
            progress_summary="",
        )

        with patch("ralph_wiggum.activities.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_tool_use = MagicMock()
            mock_tool_use.type = "tool_use"
            mock_tool_use.input = {
                "is_complete": True,
                "summary": "Addition implemented",
                "progress_update": "Completed addition",
            }
            mock_response = MagicMock()
            mock_response.content = [mock_tool_use]
            mock_client.messages.create.return_value = mock_response

            result = await evaluate_story_completion(input_data)

            assert result.is_complete is True
            assert result.summary == "Addition implemented"
            assert "Add operation" in result.updated_progress


class TestRalphWorkflow:
    """Integration tests for RalphWorkflow."""

    @pytest.mark.asyncio
    async def test_workflow_completes_single_story(self):
        """Workflow should complete when all stories are done."""

        @activity.defn(name="generate_prd")
        async def mock_generate_prd(input: GeneratePRDInput) -> PRD:
            return PRD(
                stories=[
                    Story(
                        id="story-1",
                        title="Story 1",
                        description="Do something",
                        status="pending",
                    )
                ]
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

        @activity.defn(name="evaluate_story_completion")
        async def mock_evaluate_story(input: EvaluateStoryInput) -> EvaluateStoryOutput:
            return EvaluateStoryOutput(
                is_complete=True,
                summary="Story completed",
                updated_progress="- Story 1: Done",
            )

        @activity.defn(name="evaluate_overall_completion")
        async def mock_evaluate_overall(input: EvaluateOverallInput) -> str:
            return f"All done! <promise>{input.completion_promise}</promise>"

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_generate_prd,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate_story,
                    mock_evaluate_overall,
                ],
            ):
                result = await env.client.execute_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-workflow-single-story",
                    task_queue="ralph-test",
                )

                assert result.completed is True
                assert result.completion_detected is True
                assert result.iterations_used == 1

    @pytest.mark.asyncio
    async def test_workflow_multiple_stories(self):
        """Workflow should process multiple stories sequentially."""
        story_count = [0]

        @activity.defn(name="generate_prd")
        async def mock_generate_prd(input: GeneratePRDInput) -> PRD:
            return PRD(
                stories=[
                    Story(id="1", title="Story 1", description="Desc 1", status="pending"),
                    Story(id="2", title="Story 2", description="Desc 2", status="pending"),
                ]
            )

        @activity.defn(name="generate_tasks")
        async def mock_generate_tasks(input: GenerateTasksInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return "Task output"

        @activity.defn(name="evaluate_story_completion")
        async def mock_evaluate_story(input: EvaluateStoryInput) -> EvaluateStoryOutput:
            story_count[0] += 1
            return EvaluateStoryOutput(
                is_complete=True,
                summary=f"Story {story_count[0]} done",
                updated_progress=f"- Story {story_count[0]}: Done",
            )

        @activity.defn(name="evaluate_overall_completion")
        async def mock_evaluate_overall(input: EvaluateOverallInput) -> str:
            return f"<promise>{input.completion_promise}</promise>"

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_generate_prd,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate_story,
                    mock_evaluate_overall,
                ],
            ):
                result = await env.client.execute_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-workflow-multiple-stories",
                    task_queue="ralph-test",
                )

                assert result.completed is True
                assert result.iterations_used == 2  # One iteration per story

    @pytest.mark.asyncio
    async def test_workflow_story_retry_on_incomplete(self):
        """Workflow should retry story if evaluation says incomplete."""
        eval_count = [0]

        @activity.defn(name="generate_prd")
        async def mock_generate_prd(input: GeneratePRDInput) -> PRD:
            return PRD(
                stories=[
                    Story(id="1", title="Story 1", description="Desc 1", status="pending")
                ]
            )

        @activity.defn(name="generate_tasks")
        async def mock_generate_tasks(input: GenerateTasksInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return "Task output"

        @activity.defn(name="evaluate_story_completion")
        async def mock_evaluate_story(input: EvaluateStoryInput) -> EvaluateStoryOutput:
            eval_count[0] += 1
            # First two attempts: incomplete, third: complete
            is_complete = eval_count[0] >= 3
            return EvaluateStoryOutput(
                is_complete=is_complete,
                summary="Progress" if not is_complete else "Done",
                updated_progress=f"- Attempt {eval_count[0]}",
            )

        @activity.defn(name="evaluate_overall_completion")
        async def mock_evaluate_overall(input: EvaluateOverallInput) -> str:
            return f"<promise>{input.completion_promise}</promise>"

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_generate_prd,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate_story,
                    mock_evaluate_overall,
                ],
            ):
                result = await env.client.execute_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-workflow-retry-story",
                    task_queue="ralph-test",
                )

                assert result.completed is True
                assert result.iterations_used == 3  # Retried 3 times

    @pytest.mark.asyncio
    async def test_workflow_stops_at_max_iterations(self):
        """Workflow should stop at max_iterations if stories never complete."""

        @activity.defn(name="generate_prd")
        async def mock_generate_prd(input: GeneratePRDInput) -> PRD:
            return PRD(
                stories=[
                    Story(id="1", title="Story 1", description="Desc 1", status="pending")
                ]
            )

        @activity.defn(name="generate_tasks")
        async def mock_generate_tasks(input: GenerateTasksInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return "Task output"

        @activity.defn(name="evaluate_story_completion")
        async def mock_evaluate_story(input: EvaluateStoryInput) -> EvaluateStoryOutput:
            # Always return incomplete
            return EvaluateStoryOutput(
                is_complete=False,
                summary="Still working",
                updated_progress="- Working...",
            )

        @activity.defn(name="evaluate_overall_completion")
        async def mock_evaluate_overall(input: EvaluateOverallInput) -> str:
            return "Should not be called"

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_generate_prd,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate_story,
                    mock_evaluate_overall,
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
    async def test_get_prd_query(self):
        """Should return PRD via query."""

        @activity.defn(name="generate_prd")
        async def mock_generate_prd(input: GeneratePRDInput) -> PRD:
            return PRD(
                stories=[
                    Story(id="1", title="Story 1", description="Desc 1", status="pending")
                ]
            )

        @activity.defn(name="generate_tasks")
        async def mock_generate_tasks(input: GenerateTasksInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return "Output"

        @activity.defn(name="evaluate_story_completion")
        async def mock_evaluate_story(input: EvaluateStoryInput) -> EvaluateStoryOutput:
            return EvaluateStoryOutput(
                is_complete=True,
                summary="Done",
                updated_progress="- Done",
            )

        @activity.defn(name="evaluate_overall_completion")
        async def mock_evaluate_overall(input: EvaluateOverallInput) -> str:
            return f"<promise>{input.completion_promise}</promise>"

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_generate_prd,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate_story,
                    mock_evaluate_overall,
                ],
            ):
                handle = await env.client.start_workflow(
                    RalphWorkflow.run,
                    RalphWorkflowInput(
                        prompt="Test prompt",
                        completion_promise="COMPLETE",
                        max_iterations=10,
                    ),
                    id="test-query-prd",
                    task_queue="ralph-test",
                )

                await handle.result()

                prd = await handle.query(RalphWorkflow.get_prd)
                assert prd is not None
                assert len(prd.stories) == 1
                assert prd.stories[0].status == "completed"

    @pytest.mark.asyncio
    async def test_get_progress_summary_query(self):
        """Should return progress summary via query."""

        @activity.defn(name="generate_prd")
        async def mock_generate_prd(input: GeneratePRDInput) -> PRD:
            return PRD(
                stories=[
                    Story(id="1", title="Story 1", description="Desc 1", status="pending")
                ]
            )

        @activity.defn(name="generate_tasks")
        async def mock_generate_tasks(input: GenerateTasksInput) -> list:
            return [{"content": "Task", "status": "pending"}]

        @activity.defn(name="execute_task")
        async def mock_execute_task(input: ExecuteTaskInput) -> str:
            return "Output"

        @activity.defn(name="evaluate_story_completion")
        async def mock_evaluate_story(input: EvaluateStoryInput) -> EvaluateStoryOutput:
            return EvaluateStoryOutput(
                is_complete=True,
                summary="Done",
                updated_progress="- Story 1: Completed successfully",
            )

        @activity.defn(name="evaluate_overall_completion")
        async def mock_evaluate_overall(input: EvaluateOverallInput) -> str:
            return f"<promise>{input.completion_promise}</promise>"

        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="ralph-test",
                workflows=[RalphWorkflow],
                activities=[
                    mock_generate_prd,
                    mock_generate_tasks,
                    mock_execute_task,
                    mock_evaluate_story,
                    mock_evaluate_overall,
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
                assert "Story 1" in progress
                assert "Completed successfully" in progress
