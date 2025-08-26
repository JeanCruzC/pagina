import argparse
import os

from website.scheduler import run_complete_optimization


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("solver_threads must be >= 1")
    return ivalue


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scheduling optimization")
    parser.add_argument("excel", help="Path to Excel demand file")
    parser.add_argument(
        "--solver_threads",
        type=positive_int,
        default=os.cpu_count() or 1,
        help="Number of CBC solver threads",
    )
    args = parser.parse_args()
    with open(args.excel, "rb") as f:
        run_complete_optimization(f, config={"solver_threads": args.solver_threads})


if __name__ == "__main__":
    main()
