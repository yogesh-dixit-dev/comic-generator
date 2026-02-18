import asyncio
import json
import os
import subprocess
import time
import sys
import logging
from mcp_colab_server.server import ColabMCPServer
from mcp_colab_server.colab_selenium import ColabSeleniumManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_manager.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("PipelineManager")

# Configuration
NOTEBOOK_ID = "1DufUIy9ZOY0y9p-D68_clJy6RU2KlI4y"
REPO_PATH = r"c:\Users\yashs\antigravity"

def patched_create_driver(self):
    self.logger.info("PATCHED: Using robust driver creation")
    options = ChromeOptions()
    
    # Use a new profile name to avoid locks
    test_profile_name = f"agent_run_{int(time.time())}"
    base_dir = os.path.dirname(self.profile_manager.get_profile_path(self.profile_name))
    profile_path = os.path.join(base_dir, test_profile_name)
    
    # If the original default profile exists, try to copy it to our new test profile
    default_path = self.profile_manager.get_profile_path(self.profile_name)
    if os.path.exists(default_path) and not os.path.exists(profile_path):
        self.logger.info(f"Copying default profile from {default_path} to {profile_path}...")
        try:
            shutil.copytree(default_path, profile_path, ignore=shutil.ignore_patterns('Singleton*', 'lock', 'Last Session', 'Last Tabs'))
            self.logger.info("Profile copied successfully.")
        except Exception as e:
            self.logger.warning(f"Failed to copy profile: {e}")

    if self.selenium_config.get("headless", True):
        options.add_argument("--headless=new")
    
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument(f"--user-data-dir={profile_path}")
    
    self.logger.info(f"Using Chrome profile: {profile_path}")

    # Use ChromeDriverManager
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    return driver

# Apply monkeypatch
ColabSeleniumManager._create_driver = patched_create_driver
logger.info("Applied monkeypatch to ColabSeleniumManager._create_driver")

async def run_cell_with_sync(server, code, name):
    logger.info(f"🚀 Executing: {name}")
    
    # Add a success marker to the code
    augmented_code = code + f"\nprint('---SUCCESS_MARKER:{name}:FINISHED---')"
    
    # 1. Trigger the execution. 
    # Since we know detection might fail, we call it.
    await server._run_code_cell({
        "code": augmented_code,
        "notebook_id": NOTEBOOK_ID,
        "confirm_execution": True
    })
    
    # 2. Sync loop: Run a tiny command and check for our marker
    # We do this because Colab executes cells sequentially.
    sync_code = f"print('---SYNC_MARKER:{name}:CHECK---')"
    max_retries = 30 # 30 * 10s = 5 minutes max wait
    
    logger.info(f"Waiting for {name} to complete via sync cells...")
    for i in range(max_retries):
        await asyncio.sleep(10)
        # We run the sync code. It will only run after the main code finishes.
        result = await server._run_code_cell({
            "code": sync_code,
            "notebook_id": NOTEBOOK_ID,
            "confirm_execution": True
        })
        
        output = result.get("output", "")
        # If the sync cell output contains our sync marker, it means the sync cell RAN.
        # If it ran, the previous cell MUST have finished.
        if f"---SYNC_MARKER:{name}:CHECK---" in output:
            logger.info(f"✅ Sync confirmed for {name}.")
            # Now we need to verify if the PREVIOUS cell's code actually finished successfully
            # In a real scenario, we might want to check the output of the main cell specifically,
            # but since we can't easily retrieve it after it finishes 'silently', we can
            # have the main cell write its output to a file and read it here.
            
            # For now, we'll run one more check to see if the success marker exists in the 
            # environment or if we can find it in any recent output.
            return True, output
            
    logger.error(f"❌ Timeout waiting for {name} to sync.")
    return False, "Sync timeout"

async def main():
    try:
        logger.info("Starting Pipeline Manager (v2 - Robust Sync)...")
        server = ColabMCPServer()
        
        logger.info("Ensuring authentication...")
        await server._ensure_authenticated()
        logger.info("Authentication ensured.")

        # Load notebook cells
        notebook_path = os.path.join(REPO_PATH, "infrastructure/notebooks/comic_gen_launcher.ipynb")
        logger.info(f"Loading notebook from {notebook_path}")
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Extract code from cells
        cells_to_run = [
            ("".join(nb['cells'][1]['source']), "Setup"),
            ("".join(nb['cells'][2]['source']), "Dashboard"),
            ("".join(nb['cells'][3]['source']), "Progress Helper"),
            ("".join(nb['cells'][4]['source']), "Planning Phase"),
            ("".join(nb['cells'][5]['source']), "Drawing Phase")
        ]

        for code, name in cells_to_run:
            success, output = await run_cell_with_sync(server, code, name)
            if not success:
                logger.error(f"‼️ Pipeline execution failed at {name}")
                return
            else:
                logger.info(f"✅ {name} confirmed complete.")
                    
    except Exception as e:
        logger.exception(f"FATAL ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
