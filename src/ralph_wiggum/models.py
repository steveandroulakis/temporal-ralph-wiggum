"""Data models for Ralph Wiggum workflow."""

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


@dataclass
class Story:
    """A high-level work item in the PRD."""

    id: str  # "story-1", "story-2", etc.
    title: str
    description: str
    status: str = "pending"  # pending | in_progress | completed
    completion_summary: str = ""  # Filled when marked complete


@dataclass
class PRD:
    """Product Requirements Document - list of stories."""

    stories: list[Story] = field(default_factory=list)

    def get_next_incomplete(self) -> Optional[Story]:
        """Get the next story that isn't completed."""
        for s in self.stories:
            if s.status != "completed":
                return s
        return None

    def all_complete(self) -> bool:
        """Check if all stories are completed."""
        return len(self.stories) > 0 and all(
            s.status == "completed" for s in self.stories
        )


@dataclass
class RalphWorkflowInput:
    """Input for the Ralph Wiggum workflow."""

    prompt: str
    completion_promise: str
    max_iterations: int = 20
    model: str = DEFAULT_MODEL
    history_window_size: int = 3  # Rolling window for context
    # State for continue-as-new
    current_iteration: int = 0
    conversation_history: list = field(default_factory=list)
    prd: Optional[PRD] = None  # PRD state for continue-as-new
    progress_summary: str = ""  # Rolling progress summary


@dataclass
class RalphWorkflowOutput:
    """Output from the Ralph Wiggum workflow."""

    completed: bool
    iterations_used: int
    final_response: str
    completion_detected: bool


@dataclass
class TodoItem:
    """A single task in the plan for a story."""

    content: str
    status: str = "pending"  # pending | in_progress | completed


# Activity inputs


@dataclass
class GeneratePRDInput:
    """Input for the generate_prd activity."""

    prompt: str
    model: str


@dataclass
class GenerateTasksInput:
    """Input for the generate_tasks activity (tasks for ONE story)."""

    story: Story
    original_prompt: str
    progress_summary: str
    iteration: int


@dataclass
class ExecuteTaskInput:
    """Input for the execute_task activity."""

    task_content: str
    original_prompt: str
    story_context: str  # Story title and description
    history: list  # Rolling history window
    model: str
    iteration: int
    task_index: int
    total_tasks: int


@dataclass
class EvaluateStoryInput:
    """Input for the evaluate_story_completion activity."""

    story: Story
    task_outputs: list[str]  # All outputs from this iteration
    original_prompt: str
    progress_summary: str


@dataclass
class EvaluateStoryOutput:
    """Output from the evaluate_story_completion activity."""

    is_complete: bool
    summary: str  # What was accomplished
    updated_progress: str  # New progress_summary


@dataclass
class EvaluateOverallInput:
    """Input for the evaluate_overall_completion activity."""

    prd: PRD
    original_prompt: str
    completion_promise: str


# Legacy inputs (kept for backward compatibility during transition)


@dataclass
class CallClaudeInput:
    """Input for the call_claude activity (legacy)."""

    prompt: str
    history: list
    model: str
    iteration: int
    completion_promise: str
