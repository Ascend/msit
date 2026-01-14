import argparse
import sys


def check_positive_integer(value):
    try:
        value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid integer value: %r", value) from None
    if value <= 0:
        raise argparse.ArgumentTypeError("%r is not a positive integer", value)

    return value


def confirm_continue_without_upgrade():
    RED = "\033[31m"
    RESET = "\033[0m"
    prompt_info = f"""{RED}
    This tool has matured and is now branching out as a fully independent project: msModeling
    Migration Time: January 12, 2026
    New repository address: https://gitcode.com/Ascend/msmodeling
    You can access the latest features and updates in the new repository.
    {RESET}"""
    print(prompt_info)
    user_input = input(
        f"{RED}Would you like to continue using the current version without upgrading?(y/[n]) {RESET}"
    )
    user_choice = user_input.strip().lower() or "n"

    if user_choice != "y":
        sys.exit(0)
