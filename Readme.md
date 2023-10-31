# English-Hinglish - Translator

- Completed the task with 2 different methods
1) Using Open-Ai's API
2) Tradional ML Model Trained Using Dataset

- Found A Dataset online for English To Hinglish at [https://huggingface.co/datasets/findnitai/english-to-hinglish/tree/main](English-to-Hinglish)

## Open-Ai approach
1. Clone this repository:
    ```bash
    git clone https://github.com/Delicate-Jerk/English-Hinglish.git
    cd english-to-hinglish
    ```

2. Install the required Python packages:
    ```bash
    pip install openai gradio
    ```

3. Set up your OpenAI API key:
    - Replace the `api_key` variable in the `openai-src` file with your OpenAI API key.

### Running the Translator
1. To run the translator in the terminal:
    - Uncomment the code in `openai-src.py` within the section marked for terminal use.
    - Execute the script using:
        ```bash
        python openai-src.py
        ```

2. Using the Graphical User Interface (GUI):
    - Launch the graphical interface by executing the following command in the terminal:
        ```bash
        python openai-src.py
        ```
    - This will open a web browser window where you can input English text and get the corresponding Hinglish translation.

### Input and Output
- The code allows you to input English text and receive the translated Hinglish text either through the terminal or a graphical user interface.

## For the Tradional ML Approach
- Just install the latest version of required dependiciies and run the code by replacing the path to the dataset with `hinglish_upload_v1.json`
        ```bash
        python ml-src.py
        ```

