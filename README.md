# LLM Playground

This project is a playground for experimenting with Large Language Models (LLMs).

## Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

- Python 3.12 #vllm not support 3.13
- Git

### Installation and Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/xdien/llm-playground.git
    cd llm-playground
    ```

2.  **Create and Activate the Virtual Environment:**

    This project requires a virtual environment. The following commands will create a virtual environment named `.venv-vllm` in your home directory and then activate it.

    *   Create the virtual environment:
        ```bash
        python3 -m venv ~/.venv-vllm
        ```

    *   Activate the virtual environment:
        ```bash
        . ~/.venv-vllm/bin/activate
        ```
    *After activation, your command prompt should be prefixed with `(.venv-vllm)`.*

3.  **Install PyTorch for ROCm:**

    This project requires a specific version of PyTorch compatible with ROCm. Install it using the following command:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
    ```

4.  **Install Other Dependencies:**

    Install the rest of the required packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Up Pre-commit Hooks:**

    This project uses `pre-commit` to enforce code quality and standards (e.g., ensuring all comments are in English). Install the git hooks with this command:
    ```bash
    pre-commit install
    ```
    Now, the checks will run automatically on every commit.

## Usage

You can run the main script using:
```bash
python main.py
``` 