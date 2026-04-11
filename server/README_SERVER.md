# 🚨 NetPulse — Server

**Production-grade Node.js + Express simulation engine for AI-driven incident response benchmarking.**

---

## 🧠 What is NetPulse?

NetPulse is a **real-world incident response simulation environment** where AI agents diagnose and resolve failures in a distributed system.

Unlike toy environments, NetPulse models:

* cascading service failures
* noisy vs real alerts
* resource constraints (budget, time)
* multiple valid recovery strategies

👉 Designed specifically for **LLM / agent evaluation under the OpenEnv paradigm**

---

## ⚡ Key Capabilities

* 🔁 Deterministic simulation (reproducible results)
* 🧩 Multi-service dependency graph
* 🚨 Realistic alert system (noise + signal)
* 🧠 Multi-step reasoning required (not single-step fix)
* 📊 Dense reward shaping (not sparse binary)
* 🏁 Final grading system (0.0 – 1.0 score)
* 🔄 Multiple valid solution paths

---

## 🌐 API Endpoints

| Method | Endpoint  | Description                       |
| ------ | --------- | --------------------------------- |
| POST   | `/reset`  | Initialize a new task episode     |
| POST   | `/step`   | Execute one action step           |
| GET    | `/state`  | Full internal environment state   |
| GET    | `/tasks`  | List available benchmark tasks    |
| GET    | `/score`  | Current / final performance score |
| GET    | `/health` | Service health check              |

---

## 🔁 Environment Loop

NetPulse follows a strict agent interaction loop:

```text
reset() → observation → action → step() → observation → ... → done
```

Each `/step` returns:

```json
{
  "observation": {...},
  "reward": number,
  "done": boolean,
  "info": {...}
}
```

---

## 🛠 Setup & Run

```bash
cd server
cp .env.example .env
npm install
npm start
```

### Dev mode

```bash
npm run dev
```

---

## 🧪 Running Tests

```bash
npm test
```

### Coverage includes:

* Environment lifecycle
* Reward function correctness
* Grader validation
* API contract testing

---

## 🏗 Architecture Overview

```
src/
├── app.js                  # Express app config
├── server.js               # Entry point
├── config/env.js           # Env loader
├── middleware/             # Logger + error handler
├── routes/                 # API routes
├── controllers/            # Route handlers
├── engine/                 # Core simulation logic
│   ├── simulator.js        # reset() / step() orchestrator
│   ├── stateManager.js     # In-memory state
│   ├── taskLoader.js       # Task definitions
│   ├── validationEngine.js # Action validation
│   ├── observationBuilder.js # Agent view
│   ├── actionHandler.js    # Action effects
│   ├── progressionEngine.js # System evolution
│   ├── rewardEngine.js     # Reward shaping
│   ├── grader.js           # Final scoring
│   └── constants.js        # Config values
├── models/                 # Zod schemas
├── data/                   # Tasks + metadata
└── utils/                  # Helpers
```

---

## 📊 Task Design

NetPulse includes **3 benchmark tasks**:

| Task                     | Difficulty | Description                              |
| ------------------------ | ---------- | ---------------------------------------- |
| easy_signal_noise        | Easy       | Identify real failure among noisy alerts |
| medium_hidden_dependency | Medium     | Debug indirect dependency failure        |
| hard_multi_incident      | Hard       | Handle multiple simultaneous incidents   |

Each task:

* has a hidden root cause
* requires multi-step reasoning
* is evaluated via a deterministic grader

---

## 🧮 Reward System

Reward is **dense and shaped**, not binary.

### Signals include:

* ✅ system health improvement
* ✅ customer impact reduction
* ✅ diagnostic value
* ✅ root cause progress
* ❌ penalties for bad actions

👉 Encourages intelligent, efficient decision-making

---

## 🏁 Grading System

Final score: **0.0 → 1.0**

### Based on:

* system stability
* customer harm reduction
* root cause resolution
* action efficiency
* budget utilisation
* harmful action avoidance

---

## 🔒 Determinism Guarantee

* No randomness (`Math.random()` not used)
* Pure state transitions
* JSON deep cloning
* Same input → same output always

👉 Enables reproducible benchmarking

---

## ⚙️ Environment Variables

| Variable  | Default     | Description           |
| --------- | ----------- | --------------------- |
| PORT      | 7860        | Server port (HF Standard) |
| NODE_ENV  | development | Environment mode      |
| LOG_LEVEL | info        | Logging level         |
| MAX_STEPS | 50          | Max steps per episode |

---

## 🎯 Why NetPulse Matters

Most environments are:

* ❌ games
* ❌ too simple
* ❌ unrealistic

NetPulse is:

* ✅ real-world inspired
* ✅ multi-step reasoning heavy
* ✅ suitable for LLM evaluation
* ✅ aligned with OpenEnv benchmarking

---

## 🚀 Next Steps

* Add OpenEnv compliance (`openenv.yaml`)
* Build baseline inference agent
* Deploy to Hugging Face Spaces
* Evaluate LLM performance across tasks

---

## 🧑‍💻 Author

**Made with ❤️ by Chitt Hirpara**
