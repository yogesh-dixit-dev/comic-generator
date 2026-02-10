import os
import re
import json
import logging
from typing import Any, Dict, List, Optional
from src.core.agent import BaseAgent

logger = logging.getLogger(__name__)

class TelemetryAgent(BaseAgent):
    """
    Agent specialized in analyzing logs, tracing the pipeline execution,
    and identifying areas for further performance and resilience improvements.
    """
    
    def __init__(self, agent_name: str = "TelemetryAgent", config: Dict[str, Any] = None):
        super().__init__(agent_name, config)
        self.metrics = {
            "llm_stats": {},
            "stage_latency": {},
            "errors": [],
            "retries": 0
        }

    def process(self, log_path: str) -> Dict[str, Any]:
        """
        Parses the provided log file to extract performance metrics and trace the process.
        """
        self.logger.info(f"Analyzing logs from: {log_path}")
        
        if not os.path.exists(log_path):
            self.logger.error(f"Log file not found: {log_path}")
            return {"error": "Log file not found"}

        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Regex patterns for tracing
        patterns = {
            "retry": re.compile(r"Attempt (\d+)/\d+ failed: (.+)"),
            "llm_request": re.compile(r"LLM Request \(Attempt \d+/\d+\) using (.+?)\.\.\."),
            "execution_start": re.compile(r"Starting execution for (.+?)\.\.\."),
            "execution_complete": re.compile(r"Execution handling complete for (.+?)\."),
            "timestamp": re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
        }

        trace = []
        active_stages = {}

        for line in lines:
            # 1. Capture Retries
            retry_match = patterns["retry"].search(line)
            if retry_match:
                self.metrics["retries"] += 1
                self.metrics["errors"].append({
                    "attempt": retry_match.group(1),
                    "error": retry_match.group(2),
                    "context": line.strip()
                })

            # 2. Trace Latency
            ts_match = patterns["timestamp"].search(line)
            if ts_match:
                ts = ts_match.group(1)
                
                start_match = patterns["execution_start"].search(line)
                if start_match:
                    agent = start_match.group(1)
                    active_stages[agent] = ts
                
                end_match = patterns["execution_complete"].search(line)
                if end_match:
                    agent = end_match.group(1)
                    if agent in active_stages:
                        # Simple latency calculation would go here if we parsed timestamps to datetime
                        trace.append({
                            "agent": agent,
                            "start": active_stages[agent],
                            "end": ts
                        })
                        del active_stages[agent]

        report = self.generate_report(trace)
        return report

    def generate_report(self, trace: List[Dict]) -> Dict[str, Any]:
        """
        Generates a summary report with actionable insights.
        """
        report = {
            "summary": {
                "total_retries": self.metrics["retries"],
                "unique_errors": len(set(e["error"] for e in self.metrics["errors"])),
                "agents_involved": [t["agent"] for t in trace]
            },
            "tracing": trace,
            "top_bottlenecks": self._identify_bottlenecks(trace),
            "hallucination_audit": self._audit_hallucinations(),
            "actionable_improvements": self._suggest_improvements()
        }
        return report

    def _identify_bottlenecks(self, trace: List[Dict]) -> List[str]:
        # Placeholder for more complex analytics
        return ["Slowest agents: " + ", ".join(set(t["agent"] for t in trace))]

    def _audit_hallucinations(self) -> List[str]:
        hallucinations = [e["error"] for e in self.metrics["errors"] if "validation error" in e["error"].lower()]
        return hallucinations

    def _suggest_improvements(self) -> List[str]:
        suggestions = []
        if self.metrics["retries"] > 5:
            suggestions.append("High retry count detected. Consider improving system prompts or increasing model capability.")
        
        hallucinations = self._audit_hallucinations()
        if any("dialogue" in h.lower() for h in hallucinations):
            suggestions.append("Llama models often struggle with 'dialogue' lists. Consider flatter schema structures.")
            
        if not suggestions:
            suggestions.append("Performance looks stable. Consider trying SDXL Turbo for even faster visual production.")
            
        return suggestions
