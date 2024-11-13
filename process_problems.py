import argparse
import copy
import os
import random
import time
import uuid
from logging.handlers import RotatingFileHandler
import yaml
import logging
from dataclasses import dataclass, field
from typing import List
from dataclasses import dataclass, asdict
from agent.agent import Agent, LLMParams
import numpy as np
from utils.utils import extract_score_and_mutation, get_logger_file_handler
import copy
import json
from typing import List, Tuple


@dataclass
class Problem:
    """
    Dataclass for a problem statement.
    statement: the problem statement
    score: the score of the problem
    id: the unique identifier of the problem
    suggested_mutation: the suggested mutation for the problem
    """

    statement: str
    score: float = 0.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    suggested_mutation: str = ""

    def __copy__(self):
        """
        Returns a copy of the problem.
        :return:
        """
        return Problem(self.statement, self.score, self.id, self.suggested_mutation)


class ProblemProcessor:
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initializes the ProblemProcessor with the given arguments.
        :param args:
        """
        self.problem_dir: str = "problems"
        self.output_dir: str = "output"
        self.prompts_dir: str = "prompts"
        self.current_path: str = os.path.dirname(os.path.abspath(__file__))
        self._clean_workspace()

        self.seed: int = args.seed
        self.agent: Agent = Agent(args.agent, eval=False)
        self.evaluation_agent: Agent | None = Agent(args.evaluation_agent, eval=True)
        self.num_rounds: int = args.num_rounds
        self.num_problems: int = args.num_problems
        self.topk_problems: int = args.topk_problems
        self.mutate_on_start: bool = args.mutate_on_start
        self.epsilon: float = args.epsilon
        self.best_percentage: float = args.best_percentage
        self.problems: List[Problem] = []
        self.current_problems: List[Problem] = []
        self.mutated_problems: List[Problem] = []

        self.logger: logging.Logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(get_logger_file_handler("processor.log"))

        self._get_mutations()
        self._set_seed()
        self.logger.info("ProblemProcessor initialized.")

    def _set_seed(self) -> None:
        """
        Sets the seed for random operations.
        :return:
        """

        random.seed(self.seed)
        np.random.seed(self.seed)
        self.logger.info(f"ProblemProcessor initialized with seed {self.seed}")

    def _get_mutations(self) -> None:
        """
        Get the mutations from the mutations directory.
        """

        files = os.listdir(os.path.join(self.current_path, self.prompts_dir, "mutations"))
        files = [file.split(".")[0] for file in files]
        self.mutation2id: dict = {mutation: idx for idx, mutation in enumerate(files)}
        self.id2mutation: dict = {idx: mutation for idx, mutation in enumerate(files)}

    def _clean_workspace(self) -> None:
        """
        Cleans the workspace by removing the output directory.
        :return:
        """
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
        if os.path.exists("logs"):
            for file in os.listdir("logs"):
                os.remove(os.path.join("logs", file))

    def load_problems(self) -> List[Problem]:
        """
        Loads problems from problems.txt file.
        :return: problems: list of problems
        """

        problems_path = os.path.join(self.current_path, self.problem_dir, "problems.txt")
        if not os.path.exists(problems_path):
            self.logger.error(f"{problems_path} does not exist.")
            raise FileNotFoundError(f"{problems_path} does not exist.")
        try:
            with open(problems_path, "r") as file:
                problems = [Problem(statement=line.strip()) for line in file if line.strip()]
            self.logger.info(f"Loaded {len(problems)} problems from {problems_path}")
            return problems
        except Exception as e:
            self.logger.error(f"Error loading problems: {e}")
            raise e

    def select_problems(self, round: int) -> List[Problem]:
        """
        Selects the top 3/4 of the problems by score and randomly selects the rest.
        :return: selected_problems: selected problems
        """
        self.best_percentage += round * 0.1
        self.best_percentage = min(self.best_percentage, 1.0)
        problems = copy.deepcopy(self.problems)
        three_quarters: int = int(self.best_percentage * self.num_problems)
        problems.sort(key=lambda x: x.score, reverse=True)
        best_problems = problems[:three_quarters]
        random_problems = random.sample(problems[three_quarters:],
                                        k=self.num_problems - three_quarters) if self.num_problems - three_quarters > \
            0 else []
        selected_problems = best_problems + random_problems
        self.logger.info(f"Selected {len(selected_problems)} problems for mutation")
        return selected_problems

    def save_problem(self, problem: Problem, round_num: int):
        """
        Saves a problem to a file.
        :param problem: problem to save
        :param round_num: round number
        :return:
        """
        filename = f"{problem.id}_round{round_num}.json"
        output_path = os.path.join(self.current_path, self.output_dir, filename)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.logger.info(f"Created output directory: {self.output_dir}")
        try:
            with open(output_path, "w", encoding='utf-8') as file:
                json.dump(asdict(problem), file, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving problem with id {problem.id}: {e}")
        self.logger.info(f"Saved problem: {output_path}")

    def score_problem(self, problem: Problem) -> Tuple[float, str]:
        """
        Scores a problem using the evaluation agent. If an error occurs,
        returns a random score from a uniform distribution between 1 and 10.
        :param problem: problem to score
         :return: : the score of the problem and the suggested mutation (if exists)
        """

        if self.evaluation_agent is None:
            return np.random.uniform(1, 10), ""

        eval_prompt = self.load_prompt(sub_dir="evaluations",
                                       prompt_filename="evaluate.txt")
        try:
            llm_score = self.evaluation_agent(problem.statement,
                                              eval_prompt,
                                              LLMParams(temperature=0.8))
            score, suggested_mutation = extract_score_and_mutation(llm_score)
            assert isinstance(score, float), f"Score is not a float: {score}"
            if isinstance(suggested_mutation, Exception):
                self.logger.error(f"Error extracting mutation: {suggested_mutation}")
            self.logger.info(f"Scored problem {problem.id}: {score}")
            return score, suggested_mutation
        except Exception as e:
            self.logger.error(f"Error evaluating problem: {e}")
            return np.random.uniform(1, 10), ""

    def _select_mutation(self, problem) -> str:
        """
        Selects randomly a mutation from the list of mutations.
        :return: mutation name: str
        """
        if problem.suggested_mutation != "":
            z = np.random.uniform(0, 1)
            if z < self.epsilon:
                mutation_idx = np.random.choice(len(self.id2mutation))
                return self.id2mutation[mutation_idx]
            else:
                return problem.suggested_mutation.lower()
        else:
            mutation_idx = np.random.choice(len(self.id2mutation))
            return self.id2mutation[mutation_idx]

    def apply_mutation(self, problem: Problem) -> Problem:
        """
        Applies a mutation to a problem statement. If an error occurs,
        returns the original problem.
        :param problem: problem to mutate
        :return: mutated problem: Problem
        """

        # select a mutation
        mutation = self._select_mutation(problem)
        self.logger.info(f"Applying mutation {mutation} to problem {problem.id}")
        mutation_prompt = self.load_prompt(sub_dir="mutations",
                                           prompt_filename=f"{mutation}.txt")
        # mutate the problem
        try:
            mutated_statement = self.agent(problem.statement, mutation_prompt, LLMParams(temperature=1.0))
            problem.statement = mutated_statement
            self.logger.info(f"Successfully mutated problem {problem.id}")
        except Exception as e:
            self.logger.error(f"Error mutating problem: {e}")
        # score the problem
        problem.score, suggested_mutation = self.score_problem(problem)
        if isinstance(suggested_mutation, str):
            problem.suggested_mutation = suggested_mutation if suggested_mutation in self.mutation2id else ""
        return problem

    def load_prompt(self, sub_dir: str, prompt_filename: str) -> str:
        """
        Loads a prompt from a file.
        :param sub_dir: subdirectory where the prompt file is located
        :param prompt_filename: the name of the prompt file
        :return: the prompt: str
        """

        prompt_path = os.path.join(self.current_path, self.prompts_dir, sub_dir, prompt_filename)
        if not os.path.exists(prompt_path):
            self.logger.error(f"Prompt file {prompt_filename} does not exist.")
            raise FileNotFoundError(f"Prompt file {prompt_filename} does not exist.")
        try:
            with open(prompt_path, "r") as file:
                self.logger.info(f"Loaded prompt from {prompt_path}")
                return file.read().strip()
        except Exception as e:
            self.logger.error(f"Error loading prompt: {e}")
            raise e

    def update_leaderboard(self, round: int) -> None:
        """
        Updates the leaderboard with the mutated problems. Retains the top k problems.
        :return:
        """
        self.current_problems = sorted(self.current_problems, key=lambda x: x.score, reverse=True)[:self.topk_problems]
        topk_problems = {problem.id: problem for problem in self.current_problems}
        for j in range(len(self.problems)):
            if self.problems[j].id in topk_problems:
                self.problems[j] = copy.copy(topk_problems[self.problems[j].id])
            self.save_problem(self.problems[j], round)

        self.problems.sort(key=lambda x: x.score, reverse=True)
        self.logger.info(f"Updated leaderboard. Top score: {self.problems[0].score}")

    def save_leaderboard(self) -> None:
        """
        Saves the leaderboard to a YAML file.
        :return:
        """
        try:
            leaderboard = {problem.id: problem.score for problem in self.problems}
            with open("leaderboard.yaml", "w") as file:
                yaml.dump(leaderboard, file, sort_keys=False)
            self.logger.info("Leaderboard saved to leaderboard.yaml")
        except Exception as e:
            self.logger.error(f"Error saving leaderboard: {e}")
            raise e

    def process(self) -> None:
        """
        Processes the problems by selecting, mutating, scoring, and updating the leaderboard.
        :return:
        """

        self.problems = self.load_problems()
        if self.mutate_on_start:
            self.logger.info("Mutating all the problems at the start...")
            self.problems = [self.apply_mutation(problem) for problem in self.problems]

        for round_num in range(self.num_rounds):
            self.logger.info(f"Starting round {round_num + 1} of processing...")
            # select problems
            self.current_problems = self.select_problems(round_num)
            # mutate and score selected problems
            self.current_problems = [self.apply_mutation(problem) for problem in self.current_problems]
            # update and save leaderboard
            self.update_leaderboard(round=round_num + 1)
            self.save_leaderboard()
            self.logger.info(f"Completed round {round_num + 1}")
            time.sleep(3)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and mutate problem statements.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for operations")
    parser.add_argument("--agent", type=str, default="gpt-4o-mini",
                        help="AI agent to use for mutations")
    parser.add_argument("--num_rounds", type=int, default=5,
                        help="Number of processing rounds")
    parser.add_argument("--num_problems", type=int, default=7,
                        help="Number of problems to process per round")
    parser.add_argument("--mutate_on_start", type=bool, default=False,
                        help="Flag to determine if mutation should occur at the start.")
    parser.add_argument("--topk_problems", type=int, default=5,
                        help="Number of top-performing problems to retain")
    parser.add_argument("--evaluation_agent", type=str,
                        default="gpt-4o",
                        help="AI agent to use for evaluation")
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Probability of selecting a random mutation')
    parser.add_argument('--best_percentage', type=float, default=0.3,
                        help='Percentage of best problems to retain. The rest are selected randomly.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    processor = ProblemProcessor(args)
    processor.process()
