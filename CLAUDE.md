# temporal-ralph-wiggum

Temporal workflow implementing the Ralph Wiggum loop - iterative Claude API calls until task completion (promise phrase detected) or max iterations hit.

## Commands
- `pip install -e ".[dev]"` - install deps
- `temporal server start-dev` - start Temporal (prereq)
- `python -m ralph_wiggum.worker` - run worker
- `python run_workflow.py --prompt "..." --completion-promise "DONE"` - execute workflow
- `pytest tests/` - run tests

## Key Files
- `src/ralph_wiggum/workflows.py` - RalphWorkflow (main loop, continue-as-new)
- `src/ralph_wiggum/activities.py` - call_claude, check_completion
- `src/ralph_wiggum/models.py` - dataclass inputs/outputs
- `run_workflow.py` - CLI runner

## Architecture
- Completion via `<promise>PHRASE</promise>` tag detection
- Conversation history preserved across iterations
- continue-as-new for long-running workflows
- Task queue: `ralph-wiggum-queue`

## Env
- `ANTHROPIC_API_KEY` required
