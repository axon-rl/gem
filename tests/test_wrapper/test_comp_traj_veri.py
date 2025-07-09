import fire
from transformers import AutoTokenizer

import gem
from gem.tools.search_tool import SearchTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.wrappers.complete_traj_verify_wrapper import TrajVerifyWrapper
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

test_cases = [
    {'action': '<think>I need to find out where the optic nerve crosses the midline. I\'ll search for it.</think>\n\n<search> where does the optic nerve cross the midline </search>\n\n\n<information>some information\n</information>\n\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm. Now I can provide my final answer.</think>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': True, 
     'actual_reward': 1.0,
     'description': 'Perfect case: correct answer and valid format'
     }, 

     {'action': '<think>I need to find out where the optic nerve crosses the midline. I\'ll search for it.</think>\n\n<search> where does the optic nerve cross the midline </search>\n\n\n<information>some information\n\n\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm. Now I can provide my final answer.</think>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': False, 
     'actual_reward': 0.8,
     'description': 'Missing closing information tag'
     }, 

     {'action': '<think>I need to find out where the optic nerve crosses the midline. I\'ll search for it.</think>\n\n<search> where does the optic nerve cross the midline </search>\n\n\n<information>some information\n</information>\n\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm. Now I can provide my final answer.</think>\n\n<answer> {answer} </answer>',
     'is_correct': False, 
     'formatted': True, 
     'actual_reward': 0.2,
     'description': 'Wrong answer but valid format'
     }, 

     {'action': '<think>I need to find out where the optic nerve crosses the midline. I\'ll search for it.</think>\n\n<search> where does the optic nerve cross the midline </search>\n\n\n<information>some information\n\n\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm. Now I can provide my final answer.</think>\n\n<answer> {answer} </answer>',
     'is_correct': False, 
     'formatted': False, 
     'actual_reward': 0.0,
     'description': 'Wrong answer and invalid format'
     },
     
     {'action': '<search> where does the optic nerve cross the midline </search>\n\n<information>some information\n</information>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': False, 
     'actual_reward': 0.8,
     'description': 'Missing initial think tag'
     },
     
     {'action': '<think>I need to find out where the optic nerve crosses the midline.</think>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': True, 
     'actual_reward': 1.0,
     'description': 'Missing search and information tags (valid pattern)'
     },
     
     {'action': '<answer> {answer} </answer>\n\n<think>I need to find out where the optic nerve crosses the midline.</think>',
     'is_correct': True, 
     'formatted': False, 
     'actual_reward': 0.8,
     'description': 'Wrong order: answer before think'
     },
     
     {'action': '<think>I need to find out where the optic nerve crosses the midline.</think>\n\n<search> where does the optic nerve cross the midline </search>\n\n<think>Now I will check the information.</think>\n\n<information>some information\n</information>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': False, 
     'actual_reward': 0.8,
     'description': 'Extra think tag in wrong position'
     },
     
     {'action': '<think>I need to find out where the optic nerve crosses the midline.</think>\n\nSome extra content here\n\n<search> where does the optic nerve cross the midline </search>\n\n<information>some information\n</information>\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm.</think>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': False, 
     'actual_reward': 0.8,
     'description': 'Extra content between tags'
     },
     
     {'action': '<think>I need to find out where the optic nerve crosses the midline.</think>\n\n<search> where does the optic nerve cross the midline </search>\n\n<information>some information\n</information>\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm.</think>',
     'is_correct': False, 
     'formatted': False, 
     'actual_reward': 0.0,
     'description': 'Missing answer tag (incomplete sequence)'
     },
     
     {'action': '<think>I need to find out where the optic nerve crosses the midline.\n\n<search> where does the optic nerve cross the midline </search>\n\n<information>some information\n</information>\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm.</think>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': False, 
     'actual_reward': 0.8,
     'description': 'Missing closing think tag'
     },
     
     {'action': '<think>I need to find out where the optic nerve crosses the midline.</think>\n\n<search> where does the optic nerve cross the midline \n\n<information>some information\n</information>\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm.</think>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': False, 
     'actual_reward': 0.8,
     'description': 'Missing closing search tag'
     },
     
     {'action': '<think>I need to find out where the optic nerve crosses the midline.</think>\n\n<search> where does the optic nerve cross the midline </search>\n\n<search> additional search </search>\n\n<information>some information\n</information>\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm.</think>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': False, 
     'actual_reward': 0.8,
     'description': 'Multiple search tags in wrong state'
     },
     
     {'action': '<think>I need to find out where the optic nerve crosses the midline.</think>\n\n<search> where does the optic nerve cross the midline </search>\n\n<information>some information\n</information>\n\n<search> additional search </search>\n\n<information>more information\n</information>\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm.</think>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': False, 
     'actual_reward': 0.8,
     'description': 'Multiple search cycles (valid pattern)'
     },
     
     {'action': '<think></think>\n\n<search> where does the optic nerve cross the midline </search>\n\n<information>some information\n</information>\n\n<think>I found out that the optic nerve crosses the midline at the optic chiasm.</think>\n\n<answer> {answer} </answer>',
     'is_correct': True, 
     'formatted': True, 
     'actual_reward': 1.0,
     'description': 'Empty think tag content'
     },
]


def test_single_action(search_url: str, env_name: str = "eval:QaOpen", tokenizer_name: str = "Qwen/Qwen2.5-1.5B"):
    env = gem.make(env_name, max_turns=4, unformatted_penalty=0.0, formatted_reward=0.0, is_correct_reward=1.0)
    tool = SearchTool(search_url=search_url, topk=2)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=0)
    wrapped_env = WRAPPER_FACTORY["concat_chat_on_reset"](wrapped_env, tokenizer=tokenizer)
    wrapped_env = TrajVerifyWrapper(wrapped_env, formatted_reward=0.2, verbose=True)

    print(f"Using real requests with URL: {search_url}")

    for i, test_case in enumerate(test_cases):
        print(f"------ Test {i} ------")
        print(f"Description: {test_case['description']}")
        print(f"setup: is_correct={test_case['is_correct']}, formatted={test_case['formatted']}")

        obs, info = wrapped_env.reset()
        if test_case['is_correct']:
            test_action = test_case['action'].format(answer=info['answer'][0])
        else:
            test_action = test_case['action'].format(answer='wrong answer')
        try:
            obs, reward, terminated, truncated, info = wrapped_env.step(test_action)
            if reward != test_case['actual_reward']:
                print(f"❌ Reward mismatch: expected {test_case['actual_reward']}, got {reward}")
                print(f"Observation: {obs}")
                print(f"Reward: {reward}")
                print(f"Terminated: {terminated}")
                print(f"Truncated: {truncated}")
                print(f"Info: {info}\n")
            else:
                print(f"✅ Reward matches: expected {test_case['actual_reward']}, got {reward}")
        except Exception as e:
            print(f"❌ Error during real request: {e}")
            print("Observation: [Error occurred]")
            print("Continuing with next test...\n")

if __name__ == "__main__":
    fire.Fire(test_single_action)