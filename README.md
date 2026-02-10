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
3.  Set up your environment variables (create a `.env` file):
    ```bash
    # Get FREE Gemini API key from: https://aistudio.google.com/apikey
    GOOGLE_API_KEY=your_gemini_key_here
    
    # Optional: Use OpenAI instead (requires paid credits)
    # OPENAI_API_KEY=your_key_here
    # LITELLM_MODEL=gpt-3.5-turbo
    
    # Other free/local options:
    # LITELLM_MODEL=groq/llama-3.1-8b-instant  # Free, very fast (needs GROQ_API_KEY)
    # LITELLM_MODEL=ollama/llama3              # Local, completely offline
    ```

## Usage

Run the main pipeline:

```bash
python src/main.py --input "path/to/your/story.txt"
```

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

