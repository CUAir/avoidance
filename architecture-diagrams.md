# CUAir Avoidance: Architecture Diagrams

A visual companion to [architecture.md](architecture.md), which stays the source of truth if the two ever disagree. Section references (§) point there.

The diagrams are Mermaid. GitHub renders them natively; in Cursor or VS Code, the "Markdown Preview Mermaid Support" extension renders them in the preview.

**Colors:** blue is simulation and learning code (JAX), green is perception and I/O (Python), orange is ArduPilot and aircraft hardware, red marks safety checks, purple is the ground station, and gray cylinders are stored data.

1. [System overview](#1-system-overview)
2. [Runtime decision loop](#2-runtime-decision-loop)
3. [Supervisor state machine](#3-supervisor-state-machine)
4. [One encounter, end to end](#4-one-encounter-end-to-end)
5. [Inside one simulator step](#5-inside-one-simulator-step)
6. [Training pipeline](#6-training-pipeline)
7. [Code layout](#7-code-layout)
8. [Validation ladder](#8-validation-ladder)

## 1. System overview

Training produces a policy offline. The runtime node runs it against live sensor data and hands approved paths to ArduPilot. In SITL, the "Aircraft" box is ArduPilot SITL plus a synthetic LiDAR, and the node runs on a laptop (§3, §13).

```mermaid
flowchart LR
    subgraph TRAIN["Training: GPU or laptop"]
        direction TB
        SCN["Scenario generator<br/>missions, encounters,<br/>wind, sensor parameters"]
        SIM["JAX simulator<br/>2,048 parallel environments"]
        TEACH["Sampling planner<br/>teacher and baseline"]
        LEARN["Imitation, then PPO"]
        EVAL["Evaluation suites<br/>IQM with 95% CIs"]
        SCN --> SIM
        SIM <-->|"observations, rewards,<br/>paths"| LEARN
        TEACH -->|"demonstrations"| LEARN
        LEARN --> EVAL
    end

    subgraph NODE["Runtime node: laptop in SITL, Pi 5 in flight"]
        direction TB
        PER["Perception<br/>tracks and range bins"]
        POL["Policy<br/>about 150k parameters"]
        RTA{"Safety monitor"}
        FALLBACK["Sampling planner<br/>fallback"]
        SUP["Supervisor<br/>MONITOR, AVOID, REJOIN"]
        ADP["MAVLink adapter"]
        PER --> POL --> RTA
        FALLBACK --> RTA
        RTA --> ADP
        SUP --> ADP
    end

    subgraph AIR["Aircraft: Atlas"]
        direction TB
        LID["SF30/D spinner<br/>and SPI bridge"]
        CUBE["Cube Orange+<br/>ArduPlane"]
    end

    GCS["Ground station<br/>Project-Emu"]

    TRAIN ==>|"policy that passes Gate A"| NODE
    AIR -->|"LiDAR packets<br/>and telemetry"| NODE
    NODE -->|"GUIDED commands"| AIR
    GCS -->|"AUTO mission"| AIR
    NODE -.->|"tracks and path,<br/>display only"| GCS

    classDef jax fill:#dbeafe,stroke:#1d4ed8,color:#0f172a
    classDef py fill:#dcfce7,stroke:#15803d,color:#0f172a
    classDef ap fill:#ffedd5,stroke:#c2410c,color:#0f172a
    classDef safety fill:#fee2e2,stroke:#b91c1c,color:#0f172a
    classDef ground fill:#ede9fe,stroke:#6d28d9,color:#0f172a
    class SCN,SIM,TEACH,LEARN,EVAL,POL,FALLBACK,SUP jax
    class PER,ADP py
    class LID,CUBE ap
    class RTA safety
    class GCS ground
```

## 2. Runtime decision loop

Every 0.2 s, perception output becomes an observation, the policy proposes a path, and the safety monitor approves it, substitutes the planner's path, or falls back to the emergency turn. The supervisor decides whether the policy acts at all (§5, §8–§10).

```mermaid
flowchart TB
    SPI["SF30/D readings<br/>azimuth, range, timestamp"]
    TEL["MAVLink telemetry<br/>attitude and position"]

    subgraph PERC["Perception: Python, 10 Hz"]
        POSE["Pose per reading,<br/>transform to world NED"]
        GND["Ground filter"]
        TRK["DBSCAN, Hungarian matching,<br/>Kalman tracks"]
        POSE --> GND --> TRK
    end

    subgraph DEC["Decision: JAX, 5 Hz"]
        OBS["Observation<br/>348 values"]
        POL["Policy"]
        PATH["Intended path<br/>48 points over 12 s"]
        RTA{"Safety monitor:<br/>policy path safe?"}
        PLN["Sampling planner<br/>about 320 candidates"]
        PSAFE{"Planner path safe?"}
        EMG["Emergency turn<br/>away from the threat"]
    end

    SUP["Supervisor<br/>MONITOR, AVOID, REJOIN"]
    ADP["MAVLink adapter"]
    CUBE["Cube Orange+<br/>ArduPlane"]

    SPI --> POSE
    TEL --> POSE
    SPI -->|"range bins"| OBS
    TRK -->|"tracks"| OBS
    OBS --> POL --> PATH --> RTA
    RTA -->|"yes"| ADP
    RTA -->|"no"| PLN --> PSAFE
    PSAFE -->|"yes"| ADP
    PSAFE -->|"no"| EMG --> ADP
    SUP -.->|"policy acts only in AVOID"| POL
    SUP --> ADP
    ADP --> CUBE

    classDef jax fill:#dbeafe,stroke:#1d4ed8,color:#0f172a
    classDef py fill:#dcfce7,stroke:#15803d,color:#0f172a
    classDef ap fill:#ffedd5,stroke:#c2410c,color:#0f172a
    classDef safety fill:#fee2e2,stroke:#b91c1c,color:#0f172a
    class OBS,POL,PATH,PLN,SUP jax
    class POSE,GND,TRK,ADP py
    class SPI,TEL,CUBE ap
    class RTA,PSAFE,EMG safety
```

## 3. Supervisor state machine

Engagement is rule-based, so the policy decides how to avoid, never whether. Inside `AVOID`, the flown path is chosen again every cycle (§5).

```mermaid
stateDiagram-v2
    direction LR
    [*] --> DISABLED
    DISABLED --> MONITOR: operator enables, health checks pass
    MONITOR --> AVOID: nominal path breaks well-clear within 12 s
    AVOID --> REJOIN: zero-offset path safe for 3 s
    REJOIN --> AVOID: new threat
    REJOIN --> MONITOR: cross-track under 10 m, course error under 15°
    MONITOR --> DISABLED: stale inputs, or operator disables
    AVOID --> DISABLED: pilot or GCS mode change, or inputs stale for 2 s
    REJOIN --> DISABLED: pilot or GCS mode change, or inputs stale for 2 s

    state AVOID {
        direction TB
        [*] --> PolicyPath
        PolicyPath --> PlannerPath: fails the safety monitor
        PlannerPath --> EmergencyTurn: also fails
        PolicyPath: Fly the policy path
        PlannerPath: Fly the planner path
        EmergencyTurn: Turn away at 90 percent of the bank limit
    }

    note right of MONITOR
        Aircraft in AUTO.
        Predicts separation every cycle.
    end note

    note right of REJOIN
        GUIDED back to zero offset,
        then next waypoint and AUTO.
    end note
```

## 4. One encounter, end to end

What happens from first detection to mission resumption. The shaded block is the `AVOID` loop, which runs at 5 Hz (§5, §13.2).

```mermaid
sequenceDiagram
    autonumber
    participant L as SF30/D
    participant P as Perception
    participant S as Supervisor
    participant N as Policy
    participant R as Safety monitor
    participant A as MAVLink adapter
    participant AP as ArduPilot
    participant G as Ground station

    Note over AP: Flying the mission in AUTO
    L->>P: Readings, continuously
    P->>S: Intruder track confirmed
    S->>S: Nominal path breaks well-clear within 12 s
    S->>A: Enter AVOID
    A->>AP: Switch to GUIDED
    rect rgba(220, 38, 38, 0.08)
        loop Every 0.2 s while in AVOID
            N->>R: Proposed path
            alt Path passes
                R->>A: Policy path
            else Path fails
                R->>A: Planner path, or emergency turn
            end
            A->>AP: Carrot target, or course, altitude, and speed
            A-->>G: Tracks and approved path
        end
    end
    S->>S: Zero-offset path safe for 3 s, enter REJOIN
    A->>AP: Rejoin profile back to zero offset
    S->>S: Cross-track under 10 m, course error under 15°
    A->>AP: Set current mission item to the next waypoint
    A->>AP: Switch to AUTO
    Note over AP: Mission resumes in AUTO
```

## 5. Inside one simulator step

One call to `step` is one decision. Everything inside compiles into a single XLA program and runs batched across environments (§7).

```mermaid
flowchart LR
    ACT["Action<br/>6 values in [-1, 1]"]

    subgraph VMAP["jax.jit and jax.vmap across 2,048 environments"]
        subgraph STEP["step: one decision, 0.2 s"]
            PATH["action_to_path<br/>PCHIP, feasibility projection"]
            subgraph SUB["lax.scan: 4 physics substeps of 0.05 s"]
                DYN["Ownship dynamics<br/>L1, roll lag, coordinated turn,<br/>altitude and speed lags, wind"]
                OBSM["Obstacles<br/>12 slots, lax.switch motion models"]
                LID["LiDAR<br/>attitude-rotated beam, analytic ray casts,<br/>detection model, noise"]
                TE["Track emulator<br/>Kalman filter per slot"]
                DYN --> LID
                OBSM --> LID
                LID --> TE
            end
            SUPV["Supervisor"]
            RWD["Reward<br/>true separation only"]
            OBSB["Observation builder<br/>348 values"]
            PATH --> DYN
            TE --> SUPV
            SUPV --> RWD
            SUPV --> OBSB
        end
    end

    OUT["obs, state, reward, done,<br/>info.agent_step"]
    ACT --> PATH
    RWD --> OUT
    OBSB --> OUT

    classDef jax fill:#dbeafe,stroke:#1d4ed8,color:#0f172a
    classDef store fill:#f1f5f9,stroke:#475569,color:#0f172a
    class PATH,DYN,OBSM,LID,TE,SUPV,RWD,OBSB jax
    class ACT,OUT store
```

## 6. Training pipeline

The sampling planner first teaches the policy by imitation, then PPO fine-tunes it. Nothing reaches SITL without passing Gate A against that same planner (§11, §12).

```mermaid
flowchart TB
    SUITES[("Scenario suites<br/>train, val, test_iid, test_ood, named")]
    DR["Domain randomization<br/>ranges widen from 25% to 100%"]
    SIM["JAX simulator<br/>jit and vmap"]
    PLAN["Sampling planner"]
    DATA[("Imitation dataset<br/>observation, planner action")]
    BC["Imitation pretraining<br/>plus 1 or 2 DAgger rounds"]
    PPO["PPO fine-tuning<br/>5 seeds, loss masked to AVOID,<br/>decaying imitation penalty"]
    CKPT[("Orbax checkpoints<br/>config and git SHA per run")]
    EVAL["Evaluation<br/>fast: track emulator<br/>full: Python tracker"]
    GATE{"Gate A<br/>no worse than the planner?"}
    POLICY_SITL["SITL flies the policy"]
    PLANNER_SITL["SITL flies the planner,<br/>policy keeps training"]

    SUITES --> SIM
    DR --> SIM
    SIM --> PLAN --> DATA --> BC --> PPO
    SIM <-->|"rollouts"| PPO
    PPO --> CKPT --> EVAL --> GATE
    GATE -->|"yes"| POLICY_SITL
    GATE -->|"no"| PLANNER_SITL
    PLANNER_SITL -.-> PPO

    classDef jax fill:#dbeafe,stroke:#1d4ed8,color:#0f172a
    classDef safety fill:#fee2e2,stroke:#b91c1c,color:#0f172a
    classDef store fill:#f1f5f9,stroke:#475569,color:#0f172a
    classDef ap fill:#ffedd5,stroke:#c2410c,color:#0f172a
    class DR,SIM,PLAN,BC,PPO,EVAL jax
    class GATE safety
    class SUITES,DATA,CKPT store
    class POLICY_SITL,PLANNER_SITL ap
```

## 7. Code layout

Packages under `src/avoidance/` in three layers. Arrows show which layers import which, and each box lists the packages it imports directly, including siblings in its own layer (§15).

```mermaid
flowchart TB
    cli["cli<br/>the avoid command"]

    subgraph APPS["Applications"]
        direction LR
        training["training<br/>imports sim, policy, planning"]
        evaluation["evaluation<br/>imports sim, perception,<br/>planning, policy, viz"]
        sitl["sitl<br/>imports sim"]
        runtime["runtime<br/>imports perception, planning,<br/>policy, sitl, hardware"]
    end

    subgraph COMPONENTS["Components"]
        direction LR
        sim["sim<br/>imports planning, policy"]
        perception
        planning
        policy
        hardware
        viz
    end

    subgraph FOUNDATION["Foundation: imported by every package"]
        direction LR
        core["core<br/>types, frames, geometry, packet codec"]
        config
    end

    cli --> APPS
    APPS --> COMPONENTS
    COMPONENTS --> FOUNDATION

    classDef jax fill:#dbeafe,stroke:#1d4ed8,color:#0f172a
    classDef py fill:#dcfce7,stroke:#15803d,color:#0f172a
    classDef store fill:#f1f5f9,stroke:#475569,color:#0f172a
    class training,evaluation,sim,planning,policy jax
    class sitl,runtime,perception,hardware,viz,cli py
    class core,config store
```

## 8. Validation ladder

Each rung has to pass before the next one runs. Fall 2026 ends at SITL; the hardware rungs are spring 2027 (§12–§14).

```mermaid
flowchart TB
    subgraph FALL["December 2026"]
        direction LR
        F1["Fast evaluation<br/>JAX simulator, track emulator"] --> F2["Full-fidelity evaluation<br/>Python tracker on simulated readings"] --> F3["ArduPilot SITL<br/>synthetic LiDAR, virtual obstacles"]
    end

    subgraph BENCH["Spring 2027: on the bench"]
        direction LR
        B1["Log replay"] --> B2["HITL<br/>Cube Orange+ with the Pi in the loop"] --> B3["SF30/D ground tests<br/>detection range, tilted ground returns"]
    end

    subgraph FLY["Spring 2027: in the air"]
        direction LR
        A1["Shadow mode<br/>runs and logs, never commands"] --> A2["Virtual intruders<br/>real aircraft, simulated obstacles"] --> A3["VTOL hover<br/>against a balloon"] --> A4["Forward flight<br/>against a balloon or static target"]
    end

    FALL --> BENCH --> FLY

    classDef jax fill:#dbeafe,stroke:#1d4ed8,color:#0f172a
    classDef ap fill:#ffedd5,stroke:#c2410c,color:#0f172a
    class F1,F2 jax
    class F3,B1,B2,B3,A1,A2,A3,A4 ap
```
