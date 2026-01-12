"""Data models for Ralph Wiggum workflow."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RalphWorkflowInput:
    """Input for the Ralph Wiggum workflow."""

    prompt: str
    completion_promise: str
    max_iterations: int = 20
    model: str = "claude-sonnet-4-5-20250514"
    # State for continue-as-new
    current_iteration: int = 0
    conversation_history: list = field(default_factory=list)


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


@dataclass
class CheckCompletionInput:
    """Input for the check_completion activity."""

    response: str
    completion_promise: str
