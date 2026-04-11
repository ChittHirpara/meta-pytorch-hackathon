# 🎙️ NetPulse Evaluation Demo Script
**"Watching the Agent Triage a Microservice Cascade in 3 Minutes."**

This script is designed for live evaluation or video demo purposes. It showcases the environment's ability to challenge an agent's reasoning.

---

## 🎬 Act 1: The Reset (The Incident Starts)
**Action:** Call `POST /reset` with `task_name: "easy_signal_noise"`.

**Narrator:** *"Notice the initial state. The system health is dropping fast. The API Gateway is firing 'High Latency' alerts. A typical naive agent would restart the Gateway immediately—but in NetPulse, that's a trap."*

---

## 🎬 Act 2: The Investigation
**Action:** Call `POST /step` with `action: "inspect_service", target: "api_gateway"`.

**Narrator:** *"The agent inspects the Gateway and sees that its internal health is 1.0, but its queue depth is 500. This tells the agent that the bottleneck is downstream. It needs to find the root cause before the budget runs out."*

---

## 🎬 Act 3: The Resolution
**Action:** Call `POST /step` with `action: "scale_service", target: "payment_worker"`.

**Narrator:** *"The agent identifies the Payment Worker as the bottleneck and adds capacity. Instantly, the Gateway queue begins to drain, customer impact drops, and the System Health recovers. This precision action earned a high reward signal."*

---

## 🎬 Act 4: The Result (Final Grading)
**Action:** Call `GET /score`.

**Narrator:** *"At the end of the episode, we look at the final grade. The agent succeeded with 80% of its budget remaining and zero wasted reboots. This results in a near-perfect benchmark score of 0.96."*

---
## 🛠️ Quick CLI Commands for Demo
```bash
# 1. Start the incident
curl -X POST http://localhost:7860/reset -d '{"task_name":"easy_signal_noise"}'

# 2. Inspect the symptom
curl -X POST http://localhost:7860/step -d '{"action":"inspect_service","target":"api_gateway"}'

# 3. Apply the fix
curl -X POST http://localhost:7860/step -d '{"action":"scale_service","target":"payment_worker"}'

# 4. Check the score
curl http://localhost:7860/score
```