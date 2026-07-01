"""
Configuration for the agentic bug-fix classification experiment.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASET_PATHS = [
    BASE_DIR.parent / "bugfix_pipeline" / "data" / "dataset1_100fix_300nonfix.csv",
    BASE_DIR.parent / "bugfix_pipeline" / "data" / "dataset2_100fix_300nonfix.csv",
]

N_FOLDS = 5
RANDOM_SEED = 42

# Model configuration
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ""))
MODEL_ID = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")

MAX_AGENT_STEPS = 12
MAX_KB_SEARCHES = 2
MAX_TOKENS_PER_STEP = 2048
TEMPERATURE = 0.0

KB_TOP_K = 1
KB_MAX_FEATURES = 30_000
DIFF_CHAR_LIMIT = 12_000
DIFF_SNIPPET_LINES = 30

AGENT_SYSTEM_PROMPT = """\
You are an expert software engineering researcher specializing in \
classifying commits as Bug-Fix or Non-Bug-Fix.

You will receive:
1. A TARGET commit to classify (message, code diff, repository).
2. Access to a knowledge base where you can find the most similar \
   previously human-labeled commit as a reference.

Your process:
- Search the knowledge base ONCE (at most twice) for the most similar reference commit.
- Study the reference and its human-verified label.
- Reason BACKWARD: figure out WHY the reference was given that label. \
  What specific evidence in its message and diff justified the label?
- Then investigate the target commit using analysis tools (diff, message, behavior).
- When you have sufficient evidence, call `submit_answer`. Do NOT keep searching \
  the knowledge base indefinitely — decide based on available evidence.

You are fully autonomous. You decide:
- What matters about the reference examples
- What additional information to gather about the target
- Which tools to call and in what order
- When you have enough evidence to decide

Definition of a bug-fix commit:
A bug-fix commit repairs an existing incorrect, failing, unsafe, or \
unintended behavior. It is NOT a bug-fix if it adds a feature, \
refactors code, updates documentation, reformats files, upgrades \
dependencies, or maintains tests without fixing a defect.

When you are ready to give your final answer, call the `submit_answer` \
tool with your label, confidence, and reasoning.\
"""
