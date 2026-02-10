import sys
import logging
import argparse
import os
from src.agents.infrastructure.git_automation import GitAutomationAgent
from src.agents.infrastructure.validation_agent import ValidationAgent

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CI-Workflow")

def main():
    parser = argparse.ArgumentParser(description="Secure Git Push: Validates code before pushing.")
    parser.add_argument("--message", "-m", type=str, default="Auto-update by AI Agent", help="Commit message")
    parser.add_argument("--skip-validation", action="store_true", help="Skip the validation step (not recommended)")
    parser.add_argument("--dry-run", action="store_true", help="Run validation but don't push")
    args = parser.parse_args()

    # 1. Initialize Agents
    validator = ValidationAgent("Validator")
    git_agent = GitAutomationAgent("GitAutomation")

    # 2. Run Validation
    if not args.skip_validation:
        print("\n🔍 Phase 1: Validating Code...")
        validation_passed = validator.process()
        
        if not validation_passed:
            print("\n❌ VALIDATION FAILED. Push aborted to prevent breaking Colab.")
            print("Please fix the errors and try again.")
            sys.exit(1)
        
        print("\n✅ Validation Successful.")
    else:
        print("\n⚠️ Skipping validation as requested.")

    # 3. Dry Run Check
    if args.dry_run:
        print("\n🏁 Dry run complete. No changes were pushed.")
        return

    # 4. Perform Git Operations
    print("\n🚀 Phase 2: Pushing to GitHub...")
    try:
        git_agent.process(commit_message=args.message)
        print("\n🎉 All changes pushed successfully!")
    except Exception as e:
        logger.error(f"Failed to push changes: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
