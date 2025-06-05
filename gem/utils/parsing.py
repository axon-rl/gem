from typing import Optional


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: Optional[str]):
    left = "\\boxed{"
    if isinstance(s, str):
        try:
            assert s[: len(left)] == left
            assert s[-1] == "}"
            return s[len(left) : -1]
        except:
            return None


def extract_last_boxed_answer(solution: str) -> Optional[str]:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    extracted_solution = last_boxed_only_string(solution)
    return remove_boxed(extracted_solution)
