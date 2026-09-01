import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ClinicalHealthcareMDP:
    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma
        self.states = ["Recovered", "Mild", "Moderate", "Severe", "Deceased"]
        self.actions = ["Observation", "Standard Meds", "Aggressive Therapy"]
        
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        
        # P[action, current_state, next_state]
        self.P = np.zeros((self.num_actions, self.num_states, self.num_states))
        # R[current_state, action, next_state]
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        self._initialize_transition_dynamics()
        self._initialize_reward_structure()

    def _initialize_transition_dynamics(self):
        # Absorbing states
        for a in range(self.num_actions):
            self.P[a, 0, 0] = 1.0  # Recovered
            self.P[a, 4, 4] = 1.0  # Deceased

        # Action 0: Observation (No Treatment)
        self.P[0, 1] = [0.30, 0.45, 0.20, 0.05, 0.00]  # Mild
        self.P[0, 2] = [0.05, 0.25, 0.40, 0.25, 0.05]  # Moderate
        self.P[0, 3] = [0.00, 0.05, 0.20, 0.45, 0.30]  # Severe

        # Action 1: Standard Medication
        self.P[1, 1] = [0.60, 0.30, 0.08, 0.02, 0.00]  # Mild
        self.P[1, 2] = [0.25, 0.45, 0.20, 0.08, 0.02]  # Moderate
        self.P[1, 3] = [0.05, 0.20, 0.40, 0.25, 0.10]  # Severe

        # Action 2: Aggressive Therapy / ICU
        self.P[2, 1] = [0.70, 0.20, 0.05, 0.03, 0.02]  # Mild (Toxicity risk)
        self.P[2, 2] = [0.55, 0.30, 0.10, 0.03, 0.02]  # Moderate
        self.P[2, 3] = [0.30, 0.35, 0.20, 0.10, 0.05]  # Severe

    def _initialize_reward_structure(self):
        state_costs = [0.0, -1.0, -5.0, -15.0, 0.0]
        action_costs = [0.0, -1.5, -5.0]
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_prime in range(self.num_states):
                    if s in [0, 4]:
                        self.R[s, a, s_prime] = 0.0
                    else:
                        base = state_costs[s] + action_costs[a]
                        if s_prime == 0:
                            self.R[s, a, s_prime] = base + 100.0  # Terminal cure bonus
                        elif s_prime == 4:
                            self.R[s, a, s_prime] = base - 100.0  # Terminal mortality penalty
                        else:
                            self.R[s, a, s_prime] = base

    def solve_value_iteration(self, theta: float = 1e-7):
        V = np.zeros(self.num_states)
        iterations = 0
        
        while True:
            delta = 0.0
            for s in range(1, 4):  # Non-terminal states
                v_old = V[s]
                q_vals = [
                    sum(self.P[a, s, s_prime] * (self.R[s, a, s_prime] + self.gamma * V[s_prime])
                        for s_prime in range(self.num_states))
                    for a in range(self.num_actions)
                ]
                V[s] = max(q_vals)
                delta = max(delta, abs(v_old - V[s]))
            iterations += 1
            if delta < theta:
                break
                
        # Policy extraction
        optimal_policy = np.zeros(self.num_states, dtype=int)
        for s in range(1, 4):
            q_vals = [
                sum(self.P[a, s, s_prime] * (self.R[s, a, s_prime] + self.gamma * V[s_prime])
                    for s_prime in range(self.num_states))
                for a in range(self.num_actions)
            ]
            optimal_policy[s] = int(np.argmax(q_vals))
            
        return V, optimal_policy, iterations

    def simulate_cohort(self, policy, n_patients: int = 2000, max_steps: int = 25, seed: int = 42):
        np.random.seed(seed)
        outcomes = {"Recovered": 0, "Deceased": 0, "Unresolved": 0}
        discounted_rewards = []
        step_lengths = []

        for _ in range(n_patients):
            s = np.random.choice([1, 2, 3])  # Initial: Mild, Moderate, or Severe
            total_discounted_reward = 0.0
            discount = 1.0
            
            for step in range(max_steps):
                if s in [0, 4]:
                    break
                a = policy[s] if isinstance(policy, (np.ndarray, list)) else policy
                s_next = np.random.choice(self.num_states, p=self.P[a, s])
                reward = self.R[s, a, s_next]
                
                total_discounted_reward += discount * reward
                discount *= self.gamma
                s = s_next
            
            discounted_rewards.append(total_discounted_reward)
            step_lengths.append(step)
            
            if s == 0:
                outcomes["Recovered"] += 1
            elif s == 4:
                outcomes["Deceased"] += 1
            else:
                outcomes["Unresolved"] += 1

        metrics = {
            "Recovery Rate (%)": (outcomes["Recovered"] / n_patients) * 100.0,
            "Mortality Rate (%)": (outcomes["Deceased"] / n_patients) * 100.0,
            "Mean Discounted Return": np.mean(discounted_rewards),
            "Std Return": np.std(discounted_rewards),
            "Average Steps to Exit": np.mean(step_lengths)
        }
        return metrics

# Multi-Armed Bandit Implementation for Clinical Action Selection
class ClinicalBanditEvaluator:
    def __init__(self, true_action_rewards=[15.0, 45.0, 70.0]):
        self.means = np.array(true_action_rewards)
        self.k = len(true_action_rewards)

    def run_epsilon_greedy(self, steps=1000, eps=0.1):
        q_est = np.zeros(self.k)
        counts = np.zeros(self.k)
        rewards = []
        for _ in range(steps):
            if np.random.rand() < eps:
                a = np.random.choice(self.k)
            else:
                a = np.argmax(q_est)
            r = np.random.normal(self.means[a], 2.0)
            counts[a] += 1
            q_est[a] += (r - q_est[a]) / counts[a]
            rewards.append(r)
        return np.cumsum(rewards)

    def run_ucb(self, steps=1000, c=1.5):
        q_est = np.zeros(self.k)
        counts = np.zeros(self.k)
        rewards = []
        for t in range(1, steps + 1):
            if 0 in counts:
                a = np.argmin(counts)
            else:
                ucb_vals = q_est + c * np.sqrt(np.log(t) / counts)
                a = np.argmax(ucb_vals)
            r = np.random.normal(self.means[a], 2.0)
            counts[a] += 1
            q_est[a] += (r - q_est[a]) / counts[a]
            rewards.append(r)
        return np.cumsum(rewards)

    def run_thompson_sampling(self, steps=1000):
        # Normal-Normal conjugate prior
        prior_means = np.zeros(self.k)
        prior_var = np.ones(self.k) * 100.0
        rewards = []
        for _ in range(steps):
            sampled_means = np.random.normal(prior_means, np.sqrt(prior_var))
            a = np.argmax(sampled_means)
            r = np.random.normal(self.means[a], 2.0)
            # Update Gaussian posterior
            noise_var = 4.0
            new_var = 1.0 / (1.0 / prior_var[a] + 1.0 / noise_var)
            prior_means[a] = new_var * (prior_means[a] / prior_var[a] + r / noise_var)
            prior_var[a] = new_var
            rewards.append(r)
        return np.cumsum(rewards)

if __name__ == "__main__":
    env = ClinicalHealthcareMDP()
    V_opt, pi_opt, iters = env.solve_value_iteration()
    
    print("================ OPTIMAL POLICY (VALUE ITERATION) ================")
    print(f"Converged in {iters} iterations.")
    for s_idx in range(1, 4):
        print(f"State: {env.states[s_idx]:<10} | Optimal Action: {env.actions[pi_opt[s_idx]]:<20} | V*(s): {V_opt[s_idx]:.3f}")

    benchmark_policies = {
        "Optimal MDP Policy": pi_opt,
        "Static Observation": [0, 0, 0, 0, 0],
        "Static Standard Meds": [1, 1, 1, 1, 1],
        "Static Aggressive": [2, 2, 2, 2, 2]
    }

    eval_data = []
    for name, pol in benchmark_policies.items():
        res = env.simulate_cohort(pol, n_patients=2000)
        res["Policy Strategy"] = name
        eval_data.append(res)

    df_eval = pd.DataFrame(eval_data).set_index("Policy Strategy")
    print("\n================ CLINICAL COHORT SIMULATION RESULTS (N=2000) ================")
    print(df_eval[["Recovery Rate (%)", "Mortality Rate (%)", "Mean Discounted Return", "Average Steps to Exit"]].to_string())