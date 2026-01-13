# temporal-ralph-wiggum

Temporal workflow implementing the Ralph Wiggum loop - an autonomous AI coding technique (named after the Simpsons character) where an AI agent is repeatedly fed the same prompt in a loop until task completion (promise phrase detected) or max iterations hit.

## Commands
- `pip install -e ".[dev]"` - install deps
- `temporal server start-dev` - start Temporal (prereq)
- `python -m ralph_wiggum.worker` - run worker
- `python run_workflow.py --prompt "..." --completion-promise "DONE"` - execute workflow
- `pytest tests/` - run tests

## Key Files
- `src/ralph_wiggum/workflows.py` - RalphWorkflow (iteration-decision loop, continue-as-new)
- `src/ralph_wiggum/activities.py` - decide_iteration_mode, generate_tasks, execute_task, evaluate_iteration_completion
- `src/ralph_wiggum/models.py` - dataclass inputs/outputs for activities
- `run_workflow.py` - CLI runner

## Architecture
- Per-iteration decision: `decide_iteration_mode` chooses single vs multi-task
- Single mode: execute one task directly
- Multi mode: generate_tasks then execute each
- Completion evaluated once per iteration via `evaluate_iteration_completion`
- Promise tag `<promise>PHRASE</promise>` emitted by evaluator when done
- Workflow state: `progress_summary`, `history` (carried in continue-as-new)
- Task queue: `ralph-wiggum-queue`

## Env
- `ANTHROPIC_API_KEY` required
