######################
# Imports
######################

from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

######################
# Parameters setup
######################

seed = 42
gamma = 0.99        # Discount factor
epsilon = 1.0       # Epsilon greedy
epsilon_min = 0.1   # Minimum epsilon greedy parameter
epsilon_max = 1.0   # Maximum epsilon greedy parameter
epsilon_interval = epsilon_max - epsilon_min    # Rate of reducing random actions
batch_size = 32
max_steps_per_episode = 10000

######################
# Setup environment in OpenAI gym
######################
env = make_atari('BreakoutNoFrameskip-v4')
# Wrap the inputs into correct form
# Stack 4 frames together
# Rescale the image inputs
env = wrap_deepmind(env, frame_stack=True, scale=True)
# Set random seed for environment
env.seed(seed)

######################
# Create models
######################

num_actions = 4

def create_q_model():
    # Input shape are decided by the wrapper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the input frames
    # Difference: increase num of filters in layer1
    # Difference: assign padding
    # Difference: assign kernel_initializer to change the way we decide the kernel weight matrix
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu", padding='same', kernel_initializer='lecun_normal')(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu", padding='same', kernel_initializer='lecun_normal')(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu", padding='same', kernel_initializer='lecun_normal')(layer2)
    
    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu", kernel_initializer='lecun_normal')(layer4)
    action = layers.Dense(num_actions, activation="linear", kernel_initializer='lecun_normal')(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions
# for Q-values which are used to make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
model_target = create_q_model()

######################
# Train
######################

# Set learning rate to appropriate number
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Setup replay buffers
action_history = []
rewards_history = []
state_history = []
state_next_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to act randomly
epsilon_random_frames = 10000
# Number of frames for exploration
epsilon_greedy_frames = 200000.0
# Maximum replay length to avoid memory problem
max_memory_length = 100000
# Update the model after 4 actions
update_after_actions = 4
# Update target network once every 2000 episode
update_target_network = 2000
# Use huber loss as loss function
loss_function = keras.losses.Huber()

# Number of episodes
num_episode = 50000

path = 'model/output_modified.txt'
f = open(path, 'w')

while True:  # Run until solved
    # Break if 210000 episode reached
    if episode_count > num_episode:
        break
    # Reset environment
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # Seperate the ones that update from the ones that don't
        check = 0
        frame_count += 1

        # Epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Act randomly in the beginning and when randomly selected
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Reduce the probability of acting randomly
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Choose the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save results in history lists
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        # Move to the next state
        state = state_next

        # Update every fourth frame and when batch size is reached
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Indices of the samples for history lists
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Sample from history lists
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

            # Predict next state with target model
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

            # Set the Q value of the last frame to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask to calculate loss on the updated Q-values only
            masks = tf.one_hot(action_sample, num_actions)

            # Watch over the loss
            with tf.GradientTape() as tape:
                check = 1
                # Train actor model with states and updated Q-values
                q_values = model(state_sample)
                # Apply masks on the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Compute loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Update towards the new Q values
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update target network less frequent
        if frame_count % update_target_network == 0:
            # Update the target network with new weights
            model_target.set_weights(model.get_weights())
            # Print reward from this frame
            print('Reward: ', running_reward, ' at frame: ' ,frame_count, ', episode: ', episode_count)

        # Replace old results with new ones
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]
        # If the episode is over(lose 1 life), break
        if done:
            break

    # Handle with episode rewards and write avg reward in the file
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 30:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)
    if check:
        f.write(str(episode_count)+', '+str(running_reward)+', '+str(float(loss)))
    else:
        f.write(str(episode_count)+', '+str(running_reward))
    f.write('\n')
    model.save('model/my_model_change.h5')

    episode_count += 1
    # If average reward > 40, task solved!
    if running_reward > 40:
        print("Solved at episode {}!".format(episode_count))
        break

f.close()
new_model = tf.keras.models.load_model('model/my_model_change.h5')
new_model.summary()