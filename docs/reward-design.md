# ⚖️ NetPulse Reward Function Design
**"Dense Trajectory Shaping for Efficient Agent Reasoning."**

In NetPulse, reward signals are not just binary outcomes. We use a **Composite Reward Function** that encourages diagnostic thoroughness while penalizing reckless behavior.

---

## 📐 The Reward Equation
At every step $t$, the reward $R_t$ is calculated as the sum of several weighted components:

$$ R_t = H_t + I_t + D_t + P_t - \Omega_t $$

Where:
- **$H_t$ (Health Improvement):** Points for restoring component-level health (0.0 to 1.0).
- **$I_t$ (Impact Reduction):** Points for reducing `customer_impact` (Primary Goal).
- **$D_t$ (Diagnostic Value):** Small rewards for using `inspect` actions for the *first* time on relevant components.
- **$P_t$ (Root Cause Progress):** Large "Breakthrough" rewards when the hidden root cause is addressed.
- **$\Omega_t$ (Penalty):** Subtractions for budget waste, no-ops, or re-cycling already healthy services.

---

## 🔍 Detailed Component Breakdown

### 1. Diagnostic Signal ($D_t$)
To prevent agents from guessing, we reward **Knowledge Acquisition**. An agent that inspects a faulty DB before restarting it gets a higher total reward than one that restarts it blindly.
*   *Note:* Repeated inspections on the same component yield zero reward to prevent infinite loops.

### 2. Operational Health ($H_t$)
We track the delta of the `system_health` variable. If an action stabilizes a service from 0.4 to 0.9, the agent receives a positive gradient immediately.

### 3. Customer-Facing Impact ($I_t$)
This is the **North Star metric**. Since the goal of an SRE is to save user experience, reducing `customer_impact` carries the highest passive weight in the trajectory.

### 4. Safety & Efficiency Penalties ($\Omega_t$)
*   **The "Reckless Reboot" Penalty:** Restarting a healthy service (Health > 0.9) incurs a harsh penalty to simulate unnecessary downtime.
*   **The "Quiet" Penalty:** Each step taken costs a tiny "Engineering Hour" penalty, forcing the agent to find the *shortest* path to recovery.

---

## 🏆 Final Grading vs. Step Reward
NetPulse makes a strict distinction between **Learning Signals** (Shaped Reward) and **Performance Score** (Final Grader).

The **Final Grader** (called at `done=true`) ignores the trajectory and strictly evaluates the **Final State**:
- Is the system stable?
- Is the root cause dead?
- How much money (budget) is left?

This ensures that "lucky but inefficient" agents score lower than "precise and knowledgeable" agents.

---
*NetPulse Reward Design Documentation - Meta x Hugging Face Hackathon*