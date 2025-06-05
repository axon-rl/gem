from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class BaseTool:
    tool_type = "base"
    def __init__(self, num_workers=1):
        self.num_workers = num_workers
    
    def execute_action(self, action):
        """
        Execute the action on the environment and return the observation.
        Args: action: The action to execute
        Returns:
            observation: The observation after executing the action
            done: Whether the trajectory is done
            valid: Whether the action is valid
        """
        raise NotImplementedError("Subclass must implement this method")

    def batch_execute_actions(self, actions):
        """
        Execute multiple actions in parallel.
        Args: actions: The list of actions
        Returns:
            observations: The list of observations
            dones: The list of done flags
            valids: The list of valid flags
        """
        with ThreadPoolExecutor(max_workers=min(self.num_workers, len(actions))) as executor:
            results = list(tqdm(executor.map(self.execute_action, actions),
                                            total=len(actions), desc=f"Executing actions using tool {self.tool_type}", 
                                            disable=False))
        observations, dones, valids = zip(*results)
        return observations, dones, valids
        