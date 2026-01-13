"""Data models for Ralph Wiggum workflow."""

from dataclasses import dataclass, field

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


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
    progress_summary: str = ""  # Rolling progress summary


@dataclass
class RalphWorkflowOutput:
    """Output from the Ralph Wiggum workflow."""

    completed: bool
    iterations_used: int
    final_response: str
    completion_detected: bool


# Iteration decision models


@dataclass
class DecideIterationInput:
    """Input for the decide_iteration_mode activity."""

    prompt: str
    progress_summary: str
    history: list
    iteration: int


@dataclass
class DecideIterationOutput:
    """Output from the decide_iteration_mode activity."""

    mode: str  # "single" | "multi"
    single_task_content: str  # only when mode == "single"
    rationale: str


@dataclass
class GenerateTasksInput:
    """Input for the generate_tasks activity."""

    prompt: str
    progress_summary: str
    history: list
    iteration: int


@dataclass
class ExecuteTaskInput:
    """Input for the execute_task activity."""

    task_content: str
    task_summary: str  # 2-3 word action summary for UI metadata
    original_prompt: str
    history: list  # Rolling history window
    model: str
    iteration: int
    task_index: int
    total_tasks: int


@dataclass
class EvaluateIterationInput:
    """Input for the evaluate_iteration_completion activity."""

    prompt: str
    progress_summary: str
    task_outputs: list[str]
    completion_promise: str


@dataclass
class EvaluateIterationOutput:
    """Output from the evaluate_iteration_completion activity."""

    updated_progress: str
    completion_detected: bool
    final_response: str
