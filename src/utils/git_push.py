import sys
import logging
import argparse
from src.agents.infrastructure.git_automation import GitAutomationAgent

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Git Push Automation Tool")
    parser.add_argument("--message", type=str, default="Auto-update by AI Agent", help="Commit message")
    args = parser.parse_args()

    agent = GitAutomationAgent("GitAutomationAgent")
    try:
        agent.process(commit_message=args.message)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
