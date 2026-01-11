import numpy as np
import random
from collections import deque

class ExperienceBuffer:
    """
    A replay buffer for storing and sampling experiences for reinforcement learning.

    This buffer stores past experiences (state, action, reward) and allows for
    random sampling of mini-batches to train the reinforcement learning agent.
    It uses a deque to efficiently manage a fixed-size memory buffer.
    """

    def __init__(self, buffer_size: int = 10000):
        """
        Initializes the ExperienceBuffer.

        Args:
            buffer_size: The maximum number of experiences to store in the buffer.
        """
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float):
        """
        Adds a new experience to the buffer.

        If the buffer is full, the oldest experience is automatically removed.

        Args:
            state: The state observed from the environment.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
        """
        experience = (state, action, reward)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples a random mini-batch of experiences from the buffer.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A tuple containing NumPy arrays for states, actions, and rewards.
            Returns empty lists if the buffer has fewer experiences than the batch size.
        """
        if len(self.buffer) < batch_size:
            # Not enough experiences to form a full batch
            return [], [], []

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards = zip(*batch)

        return np.array(states), np.array(actions), np.array(rewards)

    def __len__(self) -> int:
        """Returns the current number of experiences in the buffer."""
        return len(self.buffer)