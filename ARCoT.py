import os
import re
import json
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
import httpcore

load_dotenv()
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def strip_tags(text: str) -> str:
    return re.sub(r"<.*?>", "", text).strip()

class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'a', encoding='utf-8', buffering=1)

    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()

class VerificationState(TypedDict):
    question: str
    current_answer: str
    reasoning_history: list[str]
    judge_feedback: dict
    iteration: int
    max_retries: int
    judge_history: list[dict]

class EnhancedVerificationAgent:
    def __init__(self):
        self.client = deepseek_client
        self.gpt_client = gpt_client
        self.max_tokens = 200
        # Use a low temperature for concise and clear reasoning.
        self.temperature_qa = 0.0
        self.temperature_judge = 0.0
        self.workflow = self.build_workflow().compile()

        png_graph = self.workflow.get_graph(xray=True).draw_mermaid_png()
        with open("workflow_graph_ARCoT_1.png", "wb") as f:
            f.write(png_graph)

    def stream_llm(self, model, messages, temperature, client=None) -> str:
        full_content = ""
        retries = 0
        max_retries = 3
        client = client or self.client

        while retries <= max_retries:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=600
                )

                for chunk in completion:
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, "reasoning_content", "")
                    content = getattr(delta, "content", "")
                    token = reasoning or content
                    if token:
                        full_content += token
                        sys.stdout.write(token)
                        sys.stdout.flush()
                break

            except httpcore.RemoteProtocolError as e:
                retries += 1
                sys.stdout.write(f"\n[Retry {retries}] Streaming interrupted: {e}\n")
                time.sleep(1)
            except Exception as e:
                sys.stdout.write(f"\n[Error] Streaming failed: {e}\n")
                break

        return full_content.strip()

    def extract_answer(self, text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return match.group(1).strip() if match else "[UNKNOWN]"

    def invoke_llm(self, prompt: str) -> tuple[str, str]:
        # Instruct the deepseek LLM to produce a succinct chain-of-thought with the final answer in <answer> tags.
        messages = [
            {"role": "system", "content": (
                "You are a concise analytical problem solver. "
                "Provide a short, succinct chain-of-thought that directly supports your final answer. "
                "Conclude with your final answer enclosed in <answer> tags. "
                "Do not include any extra text."
            )},
            {"role": "user", "content": prompt}
        ]
        response = self.stream_llm("deepseek-reasoner", messages, self.temperature_qa)
        final_answer = self.extract_answer(response)
        return response, f"<answer>{final_answer}</answer>"

    def invoke_judge(self, question: str, reasoning: str, answer: str, history: list[str]) -> str:
        # Instruct the judge to output a JSON string with three fields: STATUS, DETAILED_FEEDBACK, RETAINABLE_ELEMENTS.
        system_prompt = (
            "You are a rigorous logical validator. Your task is to evaluate whether the latest chain-of-thought and final answer are correct. "
            "Examine the provided chain-of-thought history, previous answers, and judge feedback. "
            "Then output your evaluation in EXACTLY the following JSON format:\n\n"
            "{\n  \"STATUS\": \"Correct\" or \"Incorrect\",\n  \"DETAILED_FEEDBACK\": \"[Detailed feedback on mistakes and reasoning to avoid]\",\n  \"RETAINABLE_ELEMENTS\": \"[Any valid chain-of-thought elements worth retaining]\"\n}\n\n"
            "Include no additional text."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Question:\n{question}\n\n"
                f"Chain-of-Thought History and Answers:\n{history}\n\n"
                f"Latest Reasoning:\n{reasoning}\n\n"
                f"Latest Answer: {answer}"
            )}
        ]
        return self.stream_llm("gpt-4o", messages, self.temperature_judge, client=self.gpt_client)

    def generate_answer(self, state: VerificationState) -> VerificationState:
        print("\n===== Chain-of-Thought from generate_answer =====")
        prompt = f"""Question: {state['question']}

Provide a short, succinct chain-of-thought that directly leads to your final answer.
Conclude with your final answer enclosed in <answer> tags."""
        reasoning, final_answer = self.invoke_llm(prompt)
        state["current_answer"] = final_answer
        state["reasoning_history"].append(reasoning)
        state["iteration"] += 1
        print("\n===== Answer from generate_answer =====")
        print(final_answer)
        return state

    def revise_reasoning(self, state: VerificationState) -> VerificationState:
        print("\n===== Revised Chain-of-Thought from revise_reasoning =====")
        feedback = state["judge_feedback"].get("DETAILED_FEEDBACK", "No feedback")
        history_input = "\n\n".join(
            state["reasoning_history"] + [f"Judge said: {f['DETAILED_FEEDBACK']}" for f in state["judge_history"]]
        )
        prompt = f"""Previous Feedback:
{history_input}

Based on the above feedback, provide a new, succinct chain-of-thought that addresses the issues raised.
Conclude with your final answer enclosed in <answer> tags."""
        revised_reasoning, _ = self.invoke_llm(prompt)
        state["reasoning_history"].append(revised_reasoning.strip())
        return state

    def regenerate_answer(self, state: VerificationState) -> VerificationState:
        print("\n===== Chain-of-Thought from regenerate_answer =====")
        history_input = "\n\n".join(
            state["reasoning_history"] + [f"Judge said: {f['DETAILED_FEEDBACK']}" for f in state["judge_history"]]
        )
        prompt = f"""Previous Feedback:
{history_input}

Re-attempt Question: {state['question']}

Provide a fresh, succinct chain-of-thought that directly addresses the feedback.
Conclude with your final answer enclosed in <answer> tags."""
        reasoning, final_answer = self.invoke_llm(prompt)
        state["current_answer"] = final_answer
        state["reasoning_history"].append(reasoning.strip() if reasoning.strip() else reasoning.strip())
        state["iteration"] += 1
        print("\n===== Answer from regenerate_answer =====")
        print(final_answer)
        return state

    def llm_as_a_judge(self, state: VerificationState) -> VerificationState:
        print("\n===== LLM Judge's Assessment =====")
        reasoning_text = "\n\n".join(state["reasoning_history"][-2:])
        clean_answer = re.sub(r"\\boxed\{|\\\(|\\\)|\\\[|\\\]", "", state["current_answer"]).strip()
        history = [f"Reasoning: {r}" for r in state["reasoning_history"]] + [
            f"Feedback: {j['DETAILED_FEEDBACK']}" for j in state["judge_history"]
        ]
        raw_json = self.invoke_judge(state["question"], reasoning_text, clean_answer, history)

        try:
            parsed = json.loads(re.search(r"\{.*\}", raw_json, re.DOTALL).group())
        except Exception:
            parsed = {
                "STATUS": "Incorrect",
                "DETAILED_FEEDBACK": "Malformed judge response.",
                "RETAINABLE_ELEMENTS": "None"
            }

        if state["current_answer"].strip() in ("", "No answer extracted."):
            parsed["STATUS"] = "Incorrect"
            parsed["DETAILED_FEEDBACK"] = "No answer was generated."

        state["judge_feedback"] = parsed
        state["reasoning_history"].append(f"[Judge Feedback]: {parsed.get('DETAILED_FEEDBACK', '')}")
        state["judge_history"].append(parsed)

        if parsed.get("STATUS") == "Correct":
            state["iteration"] = state["max_retries"]

        return state

    def build_workflow(self):
        workflow = StateGraph(VerificationState)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("llm_as_a_judge", self.llm_as_a_judge)
        workflow.add_node("revise_reasoning", self.revise_reasoning)
        workflow.add_node("regenerate_answer", self.regenerate_answer)
        workflow.set_entry_point("generate_answer")
        workflow.add_edge("generate_answer", "llm_as_a_judge")

        def judge_condition(state: VerificationState):
            if state["judge_feedback"].get("STATUS") == "Correct":
                return "accept"
            if state["iteration"] >= state["max_retries"]:
                return "accept"
            return "revise"

        workflow.add_conditional_edges("llm_as_a_judge", judge_condition, {
            "accept": END,
            "revise": "revise_reasoning"
        })
        workflow.add_edge("revise_reasoning", "regenerate_answer")
        workflow.add_edge("regenerate_answer", "llm_as_a_judge")
        return workflow

    def run(self, question: str) -> VerificationState:
        return self.workflow.invoke({
            "question": question,
            "current_answer": "",
            "reasoning_history": [],
            "judge_feedback": {},
            "iteration": 0,
            "max_retries": 3,
            "judge_history": []
        })

if __name__ == "__main__":
    log_filename = f"llm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = Logger(log_filename)
    sys.stdout = logger

    try:
        agent = EnhancedVerificationAgent()
        question = "If ELI is 173, and LOIS is 5,107, how much is LESLIE?"
        result = agent.run(question)
    finally:
        sys.stdout = logger.console
        logger.close()

    print("\n===== FINAL ANSWER =====")
    print(strip_tags(result["current_answer"]))
    print("========================")
    print(f"Full log saved to: {log_filename}", flush=True)