from datasets import load_dataset

# process code contests data
ds = load_dataset("deepmind/code_contests")


def merge_columns(example):
    merged_input = (
        example["public_tests"]["input"]
        + example["private_tests"]["input"]
        + example["generated_tests"]["input"]
    )
    merged_output = (
        example["public_tests"]["output"]
        + example["private_tests"]["output"]
        + example["generated_tests"]["output"]
    )
    return {"tests": {"inputs": merged_input, "outputs": merged_output}}


ds = ds.map(merge_columns)
ds = ds.select_columns(["tests", "description"])
ds = ds.rename_columns(
    {
        "description": "new_name_1",
        "description": "problem",
    }
)
ds["test"]
