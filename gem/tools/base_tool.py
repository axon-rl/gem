class BaseTool:
    tool_type = "base"

    def __init__(self, num_workers=1):
        self.num_workers = num_workers

    def instruction_string(self) -> str:
        """
        Return the instruction string for the tool.
        This string is used to guide the agent on how to use the tool.
        Returns: Instruction string
        """
        raise NotImplementedError("Subclass must implement this method")

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
