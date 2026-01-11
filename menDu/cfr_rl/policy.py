import tensorflow as tf
import numpy as np
import config
import os

class Policy(tf.keras.Model):
    def __init__(self, state_size, action_size, learning_rate, model_path):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path  # base path for weights and full model

        # Define layers
        self.layer1 = tf.keras.layers.Dense(
            128, 
            activation='relu', 
            name="layer1", 
            input_shape=(self.state_size,) # <-- Add this line!
        )
        self.layer2 = tf.keras.layers.Dense(128, activation='relu', name="layer2")
        self.action_logits_layer = tf.keras.layers.Dense(action_size, name="action_logits")

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Build model once to initialize weights
        # Build model once to initialize weights
        dummy_input = tf.zeros((1, self.state_size))
        _ = self(dummy_input)

    def call(self, state):
        """Forward pass"""
        x = self.layer1(state)
        x = self.layer2(x)
        logits = self.action_logits_layer(x)
        actions_distribution = tf.nn.softmax(logits)
        return actions_distribution, logits

    # def predict(self, state: np.ndarray) -> np.ndarray:
    #     """Predict the action probabilities given a state"""
    #     state_batch = np.expand_dims(state, axis=0).astype(np.float32)
    #     probs, _ = self(state_batch)
    #     return probs.numpy()[0]

    def predict(self, state: np.ndarray) -> list[int]:
        """
        Predicts the top K critical endpoint indices from the state vector.
        
        Returns: A list of integer indices (e.g., [0, 2])
        """
        # 1. Prepare input: Convert to Tensor and add batch dimension (1, state_size)
        state = tf.cast(state, dtype=tf.float32)
        state_batch = tf.expand_dims(state, 0) # Shape (1, 12)

        # 2. RUN FULL FORWARD PASS (calls Policy.call)
        # This executes: state -> layer1 -> layer2 -> action_logits_layer
        _, action_logits = self(state_batch) 
        
        # 3. Get the indices of the top K largest logits
        # action_logits shape is (1, ACTION_SIZE) e.g., (1, 3)
        top_k_indices = tf.argsort(action_logits, direction='DESCENDING')[0, :config.K_CRITICAL_ENDPOINTS]
        
        # 4. Convert the TensorFlow tensor to a standard Python list of integers
        return top_k_indices.numpy().tolist()

    @tf.function
    # def train_step(self, states, actions_taken, rewards):
    #     """Single REINFORCE gradient update"""
    #     with tf.GradientTape() as tape:
    #         probs, logits = self(states)
    #         neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #             logits=logits, labels=actions_taken)
    #         loss = tf.reduce_mean(neg_log_prob * rewards)
    #     grads = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    #     return loss

    def train_step_wrapper(self, states, actions, rewards):
        """Wrapper for numpy input batches"""
        states = np.array(states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        
        # IMPORTANT: action vector is a one-hot vector (e.g., [0, 1, 0]) in buffer.
        # We need the index of the 1 (e.g., 1) for sparse_softmax_cross_entropy.
        action_indices = np.argmax(actions, axis=1).astype(np.int32)
        loss = self.train_step(states, action_indices, rewards)
        return loss.numpy() if hasattr(loss, 'numpy') else float(loss)

    def train(self, states, actions, rewards):
        """Wrapper for numpy input batches"""
        states = np.array(states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        action_indices = np.argmax(actions, axis=1).astype(np.int32)
        loss = self.train_step(states, action_indices, rewards)
        return loss.numpy() if hasattr(loss, 'numpy') else float(loss)

    def save_model(self):
        """Save weights and full model"""
        print("Saving trained model...")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Save weights as a single HDF5 file
        weights_file = self.model_path + ".h5"
        self.save_weights(weights_file)
        print(f"Weights saved to: {weights_file}")

        # Save full model in SavedModel format (creates a folder)
        full_model_dir = os.path.join(os.path.dirname(self.model_path), "full_model")
        self.save(full_model_dir)
        print(f"Full model saved in folder: {full_model_dir}")

    def load_model(self):
        """Load pre-trained model if it exists"""
        weights_file = self.model_path + ".h5"
        if os.path.exists(weights_file):
            print("Loading pre-trained weights...")
            self.load_weights(weights_file)
            print("Pre-trained weights loaded successfully")
            return True
        else:
            print("No pre-trained model found, starting with random weights")
            return False
