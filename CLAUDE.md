# temporal-ralph-wiggum

Temporal workflow implementing the Ralph Wiggum loop - iterative Claude API calls until task completion (promise phrase detected) or max iterations hit.

## Commands
- `pip install -e ".[dev]"` - install deps
- `temporal server start-dev` - start Temporal (prereq)
- `python -m ralph_wiggum.worker` - run worker
- `python run_workflow.py --prompt "..." --completion-promise "DONE"` - execute workflow
- `pytest tests/` - run tests

## Key Files
- `src/ralph_wiggum/workflows.py` - RalphWorkflow (PRD-driven loop, continue-as-new)
- `src/ralph_wiggum/activities.py` - generate_prd, generate_tasks, execute_task, evaluate_story_completion, evaluate_overall_completion
- `src/ralph_wiggum/models.py` - PRD, Story, dataclass inputs/outputs
- `run_workflow.py` - CLI runner

## Architecture
- PRD-driven: stories generated from prompt, one story per iteration
- Workflow variables: `prd`, `progress_summary` (carried in continue-as-new)
- Completion evaluated at iteration end via `evaluate_story_completion`
- Promise tag `<promise>PHRASE</promise>` only emitted by `evaluate_overall_completion`
- Task queue: `ralph-wiggum-queue`

## Env
- `ANTHROPIC_API_KEY` required
