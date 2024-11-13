# Problem Processor

## Overview
The **Problem Processor** is a Python application that automates the processing and mutation of problem statements from a `problems.txt` file. The application supports multiple processing rounds, with each round involving selection, mutation, scoring, and result tracking. The main purpose of the application is to evolve problem statements using an AI agent, retaining top-performing problems across rounds.

## Project Structure

- **problems/**: Directory containing the `problems.txt` file with problem statements.
- **output/**: Directory to store processed and mutated problem files.
- **prompts/**: Directory containing prompt templates for mutations.
- **logs/**: Directory for logging processing activities.
- **utils/**: Directory containing utility functions.
- **tests/**: Directory containing tests for the Problem Processor.
- **agent/**: Directory containing the AI agent.

## Requirements

This project requires Python 3.8 or later.

Install all necessary Python packages via `pip`:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone this repository.
2. Populate `problems/problems.txt` with problem statements (one per line) or 
use the provided sample problems.
3. Create prompt templates in `prompts/mutations/` and `prompts/evaluations/` or 
use the provided templates.
4. Create your .env file with the following variables:
    - `AZURE_OPENAI_API_KEY`: Your OpenAI API key.
    - `AZURE_OPENAI_ENDPOINT`: Your OpenAI endpoint.
   
   or set them as environment variables.
5. PyCharm is recommended for running the code. If you use it, make sure to set the working directory to the project root
and to add the .env file to the run configuration.


## Command-Line Arguments

- `--seed`: Random seed for reproducibility (default: `42`).
- `--agent`: AI agent to use for mutations (e.g., `"gpt-4"`).
- `--num_rounds`: Number of rounds to process the problems (default: `5`).
- `--num_problems`: Number of problems to process per round (default: `7`).
- `--mutate_on_start`: Flag to apply mutation at the start (default: `False`).
- `--topk_problems`: Number of top problems to retain each round (default: `5`).
- `--evaluation_agent`: AI agent for evaluation (e.g., `"gpt-4o"`).
If None, the score will be randomly samped from a uniform distribution from 0 to 10.
- `--epsilon`: Probability for selecting a random mutation (default: `0.1`).
- `--best_percentage`: Starting percentage of the best problems to retain; the rest are randomly selected (default: `0.5`).


## Agent
   - The agent incorporates the AI model used for mutations and evaluations.
   - It automatically retries the request if it fails due throttling errors for a maximum of 3 times with exponential backoff.


## Problem
   - Each problem has the following structure:
     ```json
     {
       "id": 1,
       "problem_statement": "The problem statement goes here.",
       "score": 0,
       "suggested_mutation": "expand"
     }
     ```
   - The `id` is a unique identifier for the problem.
   - The `problem_statement` is the text of the problem.
   - The `score` is the evaluation score of the problem.
   - The `suggested_mutation` is the type of mutation to apply to the problem, given by the evaluation agent, 
     or "" if no mutation is suggested, the problem is not processed yet or the mutation is not supported.


## Selection Mechanism
   - This process selects `num_problems` problems for processing. From this, `best_percentage` of the best problems are selected,
the rest of `num_problems` - `best_percentage` are randomly selected from the remaining problems.
   - `best_percentage` is a float between 0 and 1 and it is increased by 0.1 each round until 1.
   - The best problems are selected based on the evaluation score.


## Mutation Mechanism
   - The mutation mechanism is based on AI prompts.
   - The mutation is selected randomly  if the `suggested_mutation` field in the problem if is empty.
   - If the `suggested_mutation` field is not empty, it samples a number from a uniform distribution from 0 to 1 and 
     applies the mutation if the number is bigger than `epsilon`. Otherwise, it applies a random mutation.
   - The mutation is applied using the AI agent and the prompt templates from `prompts/mutations/`.

## Scoring Mechanism
   - The scoring mechanism is based on the evaluation AI agent. If the `evaluation_agent` is None, the score will be randomly samped from a uniform distribution from 0 to 10.
   - The evaluation prompt is set as system prompt and the problem statement is set as user prompt.
   - The agent is requested to score the problem statement and suggest a mutation based on its evaluation.
   - The agent should follow the evaluation format described in the system prompt that takes into account
the clarity, the conciseness, the relevance and the feasibility of the problem statement.
   - The final response should be a number between 0 and 10, representing the overall score of the problem statement.
   - Based on its logic, the agent should suggest a mutation that would improve the problem statement.
   - Its response should have JSON format and contain the following fields:
     ```json
     {
       "thinking": "The agent's chain of thought.", 
       "response": "7.5",
       "modification": "expand"
     }
     ```
     

## Workflow

1. **Initialization**:
   - Loads problems from `problems.txt`.
   - Mutates the entire set of problems at the start if `mutate_on_start` is `True`.

2. **Processing Rounds**:
   - **Selection**: Selects problems for processing based on the selection mechanism.
   - **Mutation**: Mutates each selected problem using the mutation mechanism.
   - **Scoring**: Scores each mutated problem using the scoring mechanism.
   - **Leaderboard Update**: Retains top k problems of the mutated problems based on the evaluation score and updates and saves the leaderboard.

3. **Logging and Output**:
   - Each round’s results are saved in `output/`.
   - Processed problems are saved as JSON, and the leaderboard is stored in `leaderboard.yaml`.

## Example

```bash
python process_problems.py --seed 6 --agent "gpt-4o-mini" --num_rounds 5 --num_problems 7 --mutate_on_start False --topk_problems 5
```

This command will run the processor  with seed 6, using the `gpt-4o-mini` agent, for 5 rounds, processing 7 problems per round, 
not mutating at the start,  retaining the top 5 mutated problems each round and the rest of the parameters set to default.

## Logging

All logs are saved in `logs/processor.log`, with information on processing steps, errors, and leaderboard updates.
Agent's logs are saved in `logs/agent.log` and `eval_agent.log` for the evaluation agent.

## Testing

- The tests are located in the `tests/` directory. 
- `agent_test.py` contains tests for the AI agent.
- `process_problems_test.py` contains tests for the Problem Processor.
- Run the tests from the PyCharm IDE ensuring that the `tests/` directory is set to the project root and the .env 
file is added to the run configuration.


## Tips for Hyper-Parameter Tuning
- The `epsilon` parameter controls the probability of selecting a random mutation. 
  A higher value increases the randomness of the mutations. Using strong models like 'gpt-4o' can lead to better results with a lower epsilon.
- The `best_percentage` parameter controls the percentage of the best problems to retain each round. 
A higher value increases the exploitation of the space, while a lower value increases the exploration.
When `mutate_on_start` is `True`, it is recommended to start with a higher value (e.g., 0.8)
because the space has been already explored and the scores are known. If `mutate_on_start` is `False`, 
it is recommended to start with a lower value (e.g., 0.5) and increase it each round for ensuring space exploration
and increase the number of rounds.


---
