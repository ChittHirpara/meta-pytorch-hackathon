# 🧩 NetPulse Task & Scenario Design
**"Engineering complexity into deterministic benchmarks."**

The power of NetPulse lies in its **layered difficulty progression**. Each task is designed to test a specific cognitive skill in an AI agent: scalability reasoning, cascading fault localization, and multi-incident prioritization.

---

## 🟢 Task 1: `easy_signal_noise`
**Objective:** Resolve a simple service bottleneck under misleading alert conditions.

*   **Failure Pattern:** **Queue Backpressure.** The `api_gateway` is healthy, but a downstream `notification_queue` is backing up due to a sudden traffic surge.
*   **The Trap:** The agent will see "Critical" alerts for the Gateway (Symptom), but the root cause is the Queue depth.
*   **Solution Path:** The agent must `scale_service` on the queue or `throttle_traffic` to allow the backlog to clear. 
*   **Evaluation Focus:** Basic signal vs. noise identification.

---

## 🟡 Task 2: `medium_hidden_dependency`
**Objective:** Identify a cascading failure where the root cause is invisible to superficial metrics.

*   **Failure Pattern:** **Latency Propagation.** A `database_replica` has high disk I/O latency. This causes the `payment_worker` to slow down, which eventually makes the `api_gateway` time out.
*   **The Trap:** The Database metrics look "okay" (Health 1.0) but its latency is high. The API Gateway looks "Broken" (Health 0.4).
*   **Solution Path:** The agent must use `inspect_dependency` to find the link and then `restart_service` or `failover` the database.
*   **Evaluation Focus:** Dependency tracing and multi-step root cause analysis.

---

## 🔴 Task 3: `hard_multi_incident`
**Objective:** Manage concurrent system failures while balancing a rapidly depleting budget.

*   **Failure Pattern:** **Concurrent Anomalies.** A `memory_leak` in the Auth Service combined with an `unauthorized_attack_spike` on the Analytics pipeline.
*   **The Trap:** Fixing one issue (e.g., restarting Auth) without addressing the spike will result in the budget being exhausted before the system stabilizes.
*   **Solution Path:** High-efficiency sequencing. The agent must `throttle_queue` on the spike first (Zero cost) and then `restart_service` on the leak.
*   **Evaluation Focus:** Resource management, prioritization, and long-term planning.

---

## 🧬 Scenario Variants (v1, v2, v3)
To prevent "Overfitting", each task includes deterministic variants.
- **v1:** Default scenario path.
- **v2:** Shifted metrics (e.g., higher initial latency).
- **v3:** Alternative service names/ID tags to ensure the agent is reading the topology map, not hardcoding service IDs.

---
*NetPulse Task Design Documentation - Meta x Hugging Face Hackathon*