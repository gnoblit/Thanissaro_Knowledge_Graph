# Pali Canon Knowledge Graph

This project uses Gemini's API to automatically build a knowledge graph from the Early Buddhist Pali Canon. The goal is to discover the key concepts (nodes) and relationships (edges) directly from the text itself, without a predefined schema.

## Workflow Overview

The pipeline operates in distinct phases:

1.  **Scraping:** Fetches sutta texts from `dhammatalks.org` and saves them as structured JSON.
2.  **Concept Extraction:** Uses an LLM to read each sutta and identify key concepts like figures, places, doctrines, and processes.
3.  **Canonicalization (Future):** Merges duplicate concepts (e.g., "The Buddha", "Gotama").
4.  **Relationship Extraction (Future):** Identifies the connections between the canonical concepts.
5.  **Graph Building (Future):** Populates a graph database (e.g., Neo4j).

## Project Structure

-   `config/`: Contains `settings.yaml` for all paths, model IDs, and LLM prompts.
-   `data/`: Stores all data artifacts, from raw scrapes to final graph components.
-   `logs/`: Contains logs of skipped or failed items during processing.
-   `src/`: The main Python source code, organized by function (`data_acquisition`, `processing`, `utils`).
-   `scripts/`: Executable scripts to run each phase of the pipeline.
-   `pyproject.toml` / `uv.lock`: Project and dependency management.

## Quick Start

This project uses `uv` for package and environment management.

1.  **Install `uv`:**
    ```bash
    # On macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # On Windows
    irm https://astral.sh/uv/install.ps1 | iex
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/pali-canon-kg.git
    cd pali-canon-kg
    ```

3.  **Create Environment & Install Dependencies:**
    `uv` can do both in one command. This creates a virtual environment named `.venv` and installs all dependencies from `pyproject.toml`, including the project itself in editable mode.
    ```bash
    uv sync
    ```

4.  **Activate the Environment:**
    ```bash
    # On macOS / Linux
    source .venv/bin/activate
    
    # On Windows
    .venv\Scripts\activate
    ```

5.  **Configure API Key:**
    *   Edit `.env` and set the `GEMINI_API_KEY` and/or `DEEPSEEK_API_KEY` to your own.

## How to Run

Execute the scripts from the project's root directory after activating the virtual environment.

1.  **Run the Scraper:**
    ```bash
    python scripts/01_run_scraping.py
    ```

2.  **Run Concept Extraction:**
    ```bash
    python scripts/02_run_concept_extraction.py
    ```

This will populate `data/01_raw/` and `data/03_kg_components/` with the initial data.