"""Data models for Ralph Wiggum workflow."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RalphWorkflowInput:
    """Input for the Ralph Wiggum workflow."""

    prompt: str
    completion_promise: str
    max_iterations: int = 20
    model: str = "claude-sonnet-4-5-20250929"
    history_window_size: int = 3  # Rolling window for context
    # State for continue-as-new
    current_iteration: int = 0
    conversation_history: list = field(default_factory=list)
    previous_plan: list = field(default_factory=list)  # For continue-as-new


@dataclass
class RalphWorkflowOutput:
    """Output from the Ralph Wiggum workflow."""

    completed: bool
    iterations_used: int
    final_response: str
    completion_detected: bool


@dataclass
class CallClaudeInput:
    """Input for the call_claude activity."""

    prompt: str
    history: list
    model: str
    iteration: int
    completion_promise: str


@dataclass
class CheckCompletionInput:
    """Input for the check_completion activity."""

    response: str
    completion_promise: str


@dataclass
class TodoItem:
    """A single task in the plan."""

    content: str
    status: str = "pending"  # pending | in_progress | completed


@dataclass
class GeneratePlanInput:
    """Input for the generate_plan activity."""

    prompt: str
    iteration: int
    history: list
    previous_plan: Optional[list] = None


@dataclass
class ExecuteTaskInput:
    """Input for the execute_task activity."""

    task_content: str
    original_prompt: str
    history: list
    model: str
    iteration: int
    task_index: int
    total_tasks: int
    completion_promise: str
