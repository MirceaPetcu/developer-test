import unittest
import os
from process_problems import ProblemProcessor, Problem, parse_arguments


class TestProblemProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = parse_arguments()
        cls.processor = ProblemProcessor(args)

    def test_init(self):
        self.assertIsInstance(self.processor, ProblemProcessor)
        self.assertEqual(self.processor.seed, 42)
        self.assertEqual(self.processor.num_rounds, 5)
        self.assertEqual(self.processor.num_problems, 7)
        self.assertEqual(self.processor.topk_problems, 5)

    def test_load_problems(self):
        problems = self.processor.load_problems()
        self.assertIsInstance(problems, list)
        self.assertTrue(all(isinstance(p, Problem) for p in problems))

    def test_select_problems(self):
        self.processor.current_problems = [Problem(statement=f"Problem {i}", score=i) for i in range(10)]
        selected = self.processor.select_problems(round=0)
        self.assertEqual(len(selected), self.processor.num_problems)

    def test_save_problem(self):
        problem = Problem(statement="Test problem")
        self.processor.save_problem(problem, 1)
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..',
                                 self.processor.output_dir,
                                 f"{problem.id}_round1.json")
        self.assertTrue(os.path.exists(file_path))
        os.remove(file_path)

    def test_score_problem(self):
        problem = Problem(statement="Given the list [1, 2, 3, 4, 5], what is the sum of the elements?")
        score, suggested_mutation = self.processor.score_problem(problem)
        print(score, suggested_mutation)
        self.assertIsInstance(score, float)
        self.assertIn(suggested_mutation, self.processor.mutation2id)

    def test_select_mutation(self):
        problem = Problem(statement="Test problem", suggested_mutation="")
        mutation = self.processor._select_mutation(problem)
        self.assertIn(mutation, self.processor.id2mutation.values())

    def test_apply_mutation(self):
        problem = Problem(statement="What is the capital of France?")
        original_problem_statement = problem.statement
        mutated_problem = self.processor.apply_mutation(problem)
        self.assertIsInstance(mutated_problem, Problem)
        self.assertIsInstance(problem.score, float)
        self.assertNotEqual(original_problem_statement, mutated_problem.statement)

    def test_load_prompt(self):
        prompt = self.processor.load_prompt("evaluations", "evaluate.txt")
        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 0)

    def test_update_leaderboard(self):
        self.processor.mutated_problems = [Problem(statement=f"Problem {i}", score=i) for i in range(10)]
        self.processor.update_leaderboard(round=1)
        self.assertEqual(self.processor.problems[0].score, 9)

    def test_save_leaderboard(self):
        self.processor.leaderboard = [Problem(statement=f"Problem {i}", score=i) for i in range(5)]
        self.processor.save_leaderboard()
        self.assertTrue(os.path.exists("leaderboard.yaml"))
        os.remove("leaderboard.yaml")

    def test_process(self):
        self.processor.num_rounds = 1
        self.processor.process()
        processed_problems = self.processor.problems
        self.assertTrue(any(p.score > 0 for p in processed_problems), "No problem was scored")


class TestProblem(unittest.TestCase):
    def test_problem_init(self):
        problem = Problem(statement="Test problem")
        self.assertEqual(problem.statement, "Test problem")
        self.assertEqual(problem.score, 0.0)
        self.assertIsInstance(problem.id, str)
        self.assertEqual(problem.suggested_mutation, "")


class TestParseArguments(unittest.TestCase):
    def test_parse_arguments(self):
        args = parse_arguments()
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.agent, "gpt-4o-mini")
        self.assertEqual(args.num_rounds, 5)
        self.assertEqual(args.num_problems, 7)
        self.assertEqual(args.topk_problems, 5)
        self.assertEqual(args.evaluation_agent, "gpt-4o")
        self.assertFalse(args.mutate_on_start)
        self.assertEqual(args.best_percentage, 0.3)
        self.assertEqual(args.epsilon, 0.1)


if __name__ == '__main__':
    unittest.main()
