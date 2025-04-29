**ARCoT: Agentic Revision of Chain-of-Thought**

**License: © Deniz Askin**

⸻

📚 **Overview**

Traditional agentic workflows such as ReACT, Auto-GPT, and Self-RAG focus on revising queries or executing subplans but lack access to the internal reasoning (Chain-of-Thought, CoT) of the language model.
The release of DeepSeek-r1 has introduced open access to CoT reasoning, allowing researchers to directly revise the thought processes behind answers.

ARCoT (Agentic Revision of Chain-of-Thought) introduces a novel agentic workflow that not only critiques the outputs of an LLM but iteratively revises the chain-of-thought reasoning itself when an error is detected.

This results in a dramatically enhanced accuracy, explainability, and robustness of LLM-based systems.

⸻

🎯 **Motivation**

- **Why Revise Reasoning Instead of Prompts?**
 
Directly improving reasoning pathways addresses flaws at their source rather than relying on surface-level prompt engineering.

- **New Capabilities Unlocked by DeepSeek-r1**
The exposure of explicit CoT reasoning allows for step-wise inspection, judgment, and targeted revision — a critical advancement over black-box LLM outputs.

- **Agentic Refinement of Reasoning**
By embedding a Judge-Agent and a Revision-Agent within the workflow, ARCoT enables autonomous refinement of CoT until a correct answer is produced.

⸻

🧠 **ARCoT Architecture**

![workflow_graph_ARCoT_1](https://github.com/user-attachments/assets/74dba190-766b-4043-be07-a5a3f7776eee)

1. **Question-Answering Agent:**
Generates an initial answer alongside an explicit chain-of-thought.

2.	**LLM-as-a-Judge Agent:**
Evaluates the answer and the chain-of-thought, producing:
```bash
- Status (Correct, Incorrect, Partial)
- Detailed feedback
- Retainable elements from reasoning
- Chain-of-Thought Revision Agent:
- Revises the flawed CoT using judge feedback while retaining good elements.
```
4. **Answer Regeneration Agent:**
- Generates a new answer based on the revised CoT.
5. **Iteration:**
- Repeat the process until the answer is judged correct or the workflow halts.
  
⸻

📈 **Example Workflow Execution**
	•	**Initial Run:**
The agent answers a question incorrectly due to flawed CoT.
	•	**Judge Output:**
Identifies logical inconsistencies and suggests improvements.
	•	**Revision Cycle:**
CoT is revised, emphasizing corrections without discarding valuable reasoning steps.
	•	**Answer Regeneration:**
A more accurate and robust answer is generated.
	•	**Final Output:**
If judged Correct, the workflow terminates; otherwise, it repeats refinement.

⸻

⚙️ **Technologies Used**
```bash
- Streamlit: Interactive front-end for user interactions.
- LangChain, LangGraph: LLM chaining and agentic orchestration.
- OpenAI API, IBM Watsonx: LLM service integrations.
- PyTorch: TD3 reinforcement learning agent implementation.
- scikit-learn, scipy: HRP optimization and clustering.
- Gym: Custom multi-asset trading simulation environment.
- yfinance: Real-time and historical financial data retrieval.
```
⸻

🧩 **Installation**
```bash
pip install -r requirements.txt
```
.env file (environment variables needed):
```bash
OPENAI_API_KEY=your-openai-api-key
API_KEY=your-ibm-watsonx-api-key
WATSONX_URL=your-watsonx-endpoint-url
PROJECT_ID=your-ibm-project-id
```

⸻

▶️ Running the Application
```bash
python main.py
```
```bash
- Upload a PDF.
- Enter your research/trading question.
- Run the full agentic revision workflow.
- View revised CoT, retrained agents, and final outputs in real-time.
```
⸻

✨ **Key Innovations**
	•	CoT-Aware Critiquing and Revision:
First practical framework for autonomous chain-of-thought refinement.
	•	Feedback-Guided Self-Correction:
Agents systematically incorporate judge feedback to improve answers.
	•	Reduced Overreliance on Prompt Engineering:
Direct modification of thought processes rather than superficial prompt tweaking.
	•	Scalable Iterative Reasoning:
ARCoT workflow can be expanded to multi-agent debates and collaborative revisions.

⸻

📊 **Experimental Results**
- In verbal reasoning benchmarks (e.g., Northeastern University’s Verbal Reasoning Challenge), ARCoT outperformed standard single-run DeepSeek-r1 even with lower token budgets.
- Improved handling of multi-step reasoning tasks where single-shot LLM outputs often fail.

⸻

🚀 **Future Directions**
- Integrating Retrieval-Augmented CoT Revision.
- Applying Multi-Agent Critique and Consensus.
- Extending to Complex Reasoning Benchmarks beyond verbal logic (e.g., symbolic math, program synthesis).

⸻

ARCoT represents a paradigm shift:
Instead of trying new prompts when LLMs fail, we teach the model to revise its reasoning — just as human learners would.

⸻

Would you like me to also create a flowchart diagram illustrating this entire ARCoT workflow? 🎨 It would be perfect to include under an “Architecture” section!
