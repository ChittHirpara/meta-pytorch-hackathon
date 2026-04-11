# 📊 NetPulse: Observability NOC Dashboard
**"Real-time visual telemetry for Autonomous SRE Agents."**

This is the frontend component of the NetPulse benchmark. It provides a minimalist, high-performance Network Operations Center (NOC) interface to visualize agent actions, system health cascades, and alert statuses.

---

## 🚀 Features
*   **Live Telemetry:** Real-time visualization of `system_health` and `customer_impact` metrics.
*   **Service Topology:** Visual map of active microservices (API Gateway, DB, Payment, etc.) and their health states.
*   **Action Log:** History of actions taken by the AI agent (e.g. `restart_service`, `throttle_queue`).
*   **Alert Feed:** Stream of active system alerts (Simulated CloudWatch/Datadog logs).

---

## 🛠️ Local Development

### 1. Prerequisites
Ensure the **NetPulse Backend** is running on port `7860`.

### 2. Install & Start
```bash
npm install
npm run dev
```
The dashboard will be available at `http://localhost:5173`.

---

## 🏗️ Technical Stack
*   **Framework:** React 18 + Vite
*   **Styling:** Modern CSS (Glassmorphism & NOC Aesthetics)
*   **State:** Local polling / Context for environment sync

---
*Part of the NetPulse Incident Response Benchmark.*
