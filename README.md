# Comparative Analysis of Reinforcement Learning Algorithms in Gymnasium Environments

This project implements and evaluates three popular reinforcement learning algorithms — SAC, TD3, and DDPG — on the `Humanoid-v4` environment. By leveraging `gymnasium`, `stable-baselines3`, and custom callback functions, this project tracks and visualizes performance metrics to compare the sample efficiency and overall effectiveness of each algorithm.

## Project Setup

### Requirements
- `gymnasium` for environment simulation
- `stable-baselines3` for the rl algorithms
- `torch` for GPU acceleration
- `rliable` for RL evaluation
- `matplotlib` for plotting training metrics
- `numpy` for data manipulation
- `TensorBoard` for real-time logging

### Usage
1. Clone the repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Run the training script: `python train_compare.py`
4. View logs and performance results in the `logs/` and `videos/` directories.

## Key Components

### Environment and Training
- **Custom Video Recording**: Automates video recording every 100 episodes for model performance review.
- **Custom Callback**: Logs rewards, episode lengths, and losses, saving them after each episode.

### Performance Evaluation
- **Sample Efficiency Curve**: Visualizes average rewards over timesteps.
- **Performance Profiles**: Tracks cumulative reward proportion across episodes.
- **Pairwise Probability of Improvement**: Compares each pair of algorithms to show improvement probabilities.

### Results and Graphical Analysis

The analysis of the training results for SAC, TD3, and DDPG can be summarized based on the three main graphical outputs:

**Sample Efficiency Curve:**

The sample efficiency curve showed that SAC consistently achieved higher average rewards over the 100,000 timesteps, indicating superior learning stability and efficiency.
TD3 displayed comparable performance, although with slightly more fluctuations and a lower average reward compared to SAC.
DDPG lagged in performance, with lower average rewards and higher variability throughout training.


**Performance Profiles:**

SAC's performance profile was the steepest, indicating that it not only learned efficiently but also maintained high performance throughout training.
TD3's curve was also steep but did not match the cumulative gains of SAC, suggesting it performed well but not as robustly.
DDPG had a shallower curve, showing that it struggled to achieve the same level of cumulative reward, reinforcing its weaker performance compared to the other algorithms.

**Pairwise Probability of Improvement:**

The bar chart comparing pairwise probabilities demonstrated that SAC had a very high probability (close to 1.0) of outperforming both TD3 and DDPG, solidifying its position as the top performer.
TD3 was more likely to outperform DDPG but still showed a probability lower than that of SAC.
DDPG's probability of outperforming either SAC or TD3 was close to zero, confirming its lower comparative performance.
Inferences from the Analysis
From these graphical results, we can derive the following inferences:

SAC is the Best Choice: SAC's consistent and high average reward over multiple runs, along with its dominant position in the pairwise probability analysis, indicates that it is the most effective algorithm for training the Mujoco Humanoid-v4 model in terms of both sample efficiency and overall performance.
TD3 is Competitive but Less Robust: While TD3 shows decent performance, it does not reach the consistency or the level of SAC. It may be a viable alternative when computational resources or specific model constraints favor its use, but it might require more careful tuning.
DDPG Falls Behind: DDPG's performance was notably weaker across all metrics. It may not be well-suited for complex control tasks in environments like Mujoco's Humanoid-v4 without significant modifications or enhancements.
Overall, the comparative analysis through these graphical outputs clearly positions SAC as the most reliable and efficient algorithm for this specific application.

# Extended Implementation: SAC Algorithm Training for Humanoid-v4

Building on the findings that SAC is the most efficient and robust reinforcement learning algorithm for training the Mujoco Humanoid-v4 model, a dedicated training session was conducted using SAC to fully leverage its advantages. This extended implementation included several enhancements and tools to ensure effective training and detailed analysis.

## Key Features of the SAC Training Implementation:

**Model Saving & Loading**: The training setup included mechanisms to check for pre-existing models and continue training from the last saved checkpoint. This feature ensures the preservation of training progress, including replay buffers and tracked metrics, which contributes to consistent learning without starting from scratch.

**Custom Callbacks & Logging**: A custom callback function was developed to log rewards and other metrics during training. The rewards and training progress were logged to disk for comprehensive post-training analysis. TensorBoard was integrated to provide real-time visualization of training metrics, making it easier to monitor the agent's performance.

**Video Recording**: To assess the qualitative behavior of the trained agent, a custom video recorder was implemented, capturing episodes at regular intervals (every 100 episodes). This visual record was essential for observing how the agent's actions evolved over time and gauging its proficiency in completing tasks.

**Training Visualization**: The cumulative rewards obtained during training were plotted using matplotlib, allowing for a straightforward evaluation of the learning curve and sample efficiency over time.

Libraries and Tools Used:

-`gymnasium`: For environment simulation and interaction.
-`stable-baselines3`: Specifically for the SAC algorithm implementation, ensuring efficient learning and policy updates.
-`torch`: To leverage GPU acceleration, providing faster model training.
-`matplotlib`: For plotting training metrics, facilitating visual analysis.
-`numpy`: For data manipulation and efficient array operations.
-`TensorBoard`: For real-time logging, which helped track various metrics such as episode rewards and losses throughout the training process.

# Inference from Extended SAC Training:

The detailed implementation reinforced the earlier comparison findings by showcasing SAC’s superior capability in training complex models like Mujoco's Humanoid-v4. The logged metrics and recorded videos demonstrated SAC's stability and effectiveness in mastering the environment, validating it as the top choice for reinforcement learning applications in this domain.

## Acknowledgments
This project utilizes [rliable](https://github.com/google-research/rliable) for robust RL evaluations and is inspired by stable-baselines3 documentation and Gymnasium environments.
