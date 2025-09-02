# Transactive Cognitive Memory System for Multi-Agent AI

## Overview

Implementation framework for Transactive Cognitive Memory (TCM) in multi-agent artificial intelligence systems. This repository provides the core components for building AI teams that efficiently distribute knowledge through learned specialization patterns.

## Research Context

This framework implements the theoretical model presented in "Transactive Cognitive Memory for Multi-Agent AI and Distributed Systems" (Naik, 2025), demonstrating how artificial agents can develop expertise awareness similar to human teams.

## Core Components

### Agent Architecture

The system implements three specialized agent roles:

```python
agents = {
    "planner": PlannerAgent(),      # Task decomposition and strategy
    "researcher": ResearcherAgent(), # Information synthesis
    "verifier": VerifierAgent()      # Validation and trust updates
}
```

### Trust Evolution Framework

Trust parameters evolve through Bayesian updates using Beta distributions:

```python
@dataclass
class BetaTrust:
    alpha: float = 1.0  # Prior successes + 1
    beta: float = 1.0   # Prior failures + 1
    
    def sample(self):
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, success: bool):
        if success: 
            self.alpha += 1
        else: 
            self.beta += 1
```

### Memory Backend Implementations

| Backend | Description | Use Case |
|---------|-------------|----------|
| Isolated | Per-agent private memory | High precision requirements |
| Shared | Single global store | Maximum knowledge reuse |
| Selective | Rule-based filtering | Predefined expertise domains |
| TCM | Trust-weighted routing | Dynamic specialization |

## Algorithm Implementation

### TCM Delegation with Thompson Sampling

```python
def delegate_query(query, agents, trust_params):
    # Thompson Sampling for probability matching
    samples = {}
    for agent_id in agents:
        theta = trust_params[agent_id].sample()
        samples[agent_id] = theta
    
    # Select agent with highest sampled value
    best_agent = max(samples, key=samples.get)
    
    # Process query and update trust
    result = agents[best_agent].process(query)
    success = (result.verdict == "SUPPORTED")
    trust_params[best_agent].update(success)
    
    return result, best_agent
```

## Installation and Setup

### Requirements

- Python 3.9 or higher
- NumPy, SciPy for numerical computations
- NetworkX for knowledge graph management
- Optional: OpenAI/Anthropic API for LLM integration

### Basic Installation

```bash
git clone https://github.com/SheetalNaik98/Transactive-Cognitive-Memory-System-for-Multi-Agent-AI.git
cd Transactive-Cognitive-Memory-System-for-Multi-Agent-AI
pip install -r requirements.txt
```

## Usage Examples

### Initialize System

```python
from tcm_lab import TCMSystem

# Create system with TCM backend
system = TCMSystem(
    agents=["planner", "researcher", "verifier"],
    backend="tcm"
)

# Process query
result = system.process(
    "Design a recommendation system for e-commerce"
)

# View trust evolution
trust_scores = system.get_trust_scores()
print(trust_scores)
# Output: {'planner_planning': 0.95, 'researcher_ml': 0.82, ...}
```

### Comparative Analysis

```python
# Compare all memory backends
results = {}
for backend in ["isolated", "shared", "selective", "tcm"]:
    system = TCMSystem(backend=backend)
    metrics = system.evaluate(test_queries)
    results[backend] = metrics

# Analysis shows TCM achieves 66.7% delegation rate
```

## Experimental Framework

### Task Categories

1. Design Tasks: System architecture and planning
2. Research Tasks: Information gathering and synthesis
3. Verification Tasks: Claim validation and fact-checking
4. Implementation Tasks: Code generation and optimization
5. Analysis Tasks: Pattern recognition and data analysis

### Evaluation Metrics

```python
class MetricsCalculator:
    def delegation_rate(self, logs):
        """Primary metric for specialization efficiency"""
        delegated = sum(1 for l in logs if l.delegated)
        return delegated / len(logs)
    
    def memory_efficiency(self, logs):
        """Memory Reads per Supported Verdict (MRSV)"""
        supported = [l for l in logs if l.verdict == "SUPPORTED"]
        return sum(l.memory_reads for l in supported) / len(supported)
    
    def trust_convergence_rate(self, trust_history):
        """Iterations until trust stabilization"""
        # Implementation details in metrics.py
        pass
```

## Research Contributions

### Theoretical Advances

- First application of transactive memory theory to artificial agents
- Formal convergence proof for Beta-Bernoulli trust model
- Thompson Sampling adaptation for multi-agent delegation

### Empirical Results

- Delegation Rate: 66.7% for TCM vs 0% for baseline systems
- Task Success: 100% accuracy maintained across all backends
- Convergence: Trust parameters stabilize within 10-15 interactions
- Scalability: Linear complexity with agent count

### Implementation Contributions

- Modular framework for multi-agent systems
- Pluggable memory backend architecture
- Real-time monitoring and visualization tools
- Comprehensive evaluation suite

## Documentation

### API Reference

Detailed documentation available in `docs/api.md`

### Experiment Reproduction

Step-by-step guide in `docs/experiments.md`

### System Architecture

Technical specifications in `docs/architecture.md`

## Related Publications

### Primary References

1. Thompson, W.R. (1933). On the Likelihood That One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. Biometrika, 25, 285-294.

2. Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A Tutorial on Thompson Sampling. Foundations and Trends in Machine Learning, 11(1), 1-96.

3. Zulfikar, W., Chan, S., & Maes, P. (2024). Memoro: Using Large Language Models to Realize a Concise Interface for Real-Time Memory Augmentation. CHI '24.

4. Kirmayr, J., et al. (2025). CarMem: Enhancing Long-Term Memory in LLM Voice Assistants through Category-Bounding. arXiv:2501.09645.

5. PIN AI Team, et al. (2025). GOD model: Privacy Preserved AI School for Personal Assistant. arXiv:2502.18527.

## Project Information

### Academic Context

Master of Science Research Project  
Data Analytics Engineering  
Northeastern University  
Advisor: Dr. Mohammad Dehghani

### Repository Links

- Main Implementation: https://github.com/SheetalNaik98/TCM-for-Distributed-Systems
- Framework Repository: https://github.com/SheetalNaik98/Transactive-Cognitive-Memory-System-for-Multi-Agent-AI

## License

MIT License

## Contact

Sheetal Naik  
MS Data Analytics Engineering  
Northeastern University
