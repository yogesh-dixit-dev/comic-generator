# AI Comic Book Generator

A modular, agent-based system for converting text (novels, stories, scripts) into consistent comic book pages.

## Features
- **Multi-Format Input:** Supports `.txt`, `.pdf`, `.docx`.
- **Intelligent Scripting:** Breaks down stories into scenes and panels with director notes.
- **Character Consistency:** Manages character visual profiles across panels.
- **Cinematic Direction:** Automatically adds camera angles and lighting.
- **Modular Architecture:** Swap LLMs (OpenAI, Ollama) and Image Generators (Stable Diffusion, Midjourney) easily.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Option A: Local Ollama (NO API KEYS - Recommended)**
    ```bash
    # Install Ollama (one-time setup)
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Pull the model
    ollama pull llama3.2
    
    # That's it! No environment variables needed.
    ```

4.  **Option B: Cloud APIs (requires API keys)**
    ```bash
    # Get FREE Gemini API key from: https://aistudio.google.com/apikey
    GOOGLE_API_KEY=your_gemini_key_here
    LITELLM_MODEL=gemini/gemini-pro
    
    # OR use OpenAI (requires paid credits)
    # OPENAI_API_KEY=your_key_here
    # LITELLM_MODEL=gpt-3.5-turbo
    ```

## Usage

### 🚀 Standard Pipeline
Run the full automated pipeline:
```bash
python src/main.py --input "path/to/your/story.txt"
```

### 🎨 Phase 2: Interactive (HITL) Mode
For maximum control and VRAM efficiency (especially in Colab), use the **Interactive Dashboard**:

1. **Start the Backend Server**:
   ```bash
   $env:PYTHONPATH='.'; venv/Scripts/python.exe infrastructure/server/app.py
   ```
2. **Start the React Dashboard**:
   ```bash
   cd infrastructure/web
   npm install && npm run dev
   ```
3. **Run in Phases**:
   - **Planning (LLM)**: `python src/main.py --input story.txt --phase plan`
   - **Drawing (GPU)**: `python src/main.py --input story.txt --phase draw`

> [!TIP]
> In HITL mode, the pipeline pauses after the **Script** and **Character Design** steps. Open the dashboard at `http://localhost:3000` to review, edit, and approve before the GPU-heavy drawing begins.

## Architecture

The system uses a pipeline of specialized agents:

1.  **InputReader:** Ingests and chunks text.
2.  **ScriptWriter:** Converts prose to `ComicScript` (JSON).
3.  **ScriptCritique:** Reviews script quality.
4.  **CharacterDesigner:** Generates consistent character sheets.
5.  **Director:** Adds camera angles and lighting.
6.  **ConsistencyManager:** Ensures visual continuity.
7.  **Illustrator:** Generates the final images.

## Project Structure

- `src/core`: Base classes and data models.
- `src/agents`: Specialized agent implementations.
- `src/utils`: Helper functions (LLM interface, etc.).
- `output`: Generated scripts and images.

