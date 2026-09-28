import os

import gradio as gr
from transformers import pipeline

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# IBM WATSONX.AI CONFIGURATION
# ============================================================

WATSONX_URL = "https://us-south.ml.cloud.ibm.com"

PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
API_KEY = os.getenv("WATSONX_API_KEY")

if not PROJECT_ID or not API_KEY:
    raise ValueError(
        "Missing watsonx.ai credentials. "
        "Set WATSONX_PROJECT_ID and WATSONX_API_KEY "
        "as environment variables."
    )

credentials = Credentials(
    url=WATSONX_URL,
    api_key=API_KEY
)

model_id = "ibm/granite-4-h-small"

parameters = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1,
}


# ============================================================
# GRANITE LLM
# ============================================================

llm = WatsonxLLM(
    model_id=model_id,
    url=WATSONX_URL,
    project_id=PROJECT_ID,
    apikey=API_KEY,
    params=parameters
)


# ============================================================
# WHISPER SPEECH-TO-TEXT
# ============================================================

# Load Whisper once when the application starts.
# This avoids reloading the model every time an audio file is submitted.
speech_to_text = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    chunk_length_s=30,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def remove_non_ascii(text):
    """Remove non-ASCII characters from the transcript."""

    return "".join(
        character
        for character in text
        if ord(character) < 128
    )


def product_assistant(ascii_transcript):
    """
    Normalize financial terminology before generating
    the final meeting minutes.
    """

    system_prompt = """
    You are an intelligent assistant specializing in financial products.

    Process the meeting transcript and normalize financial terminology
    and commonly used financial acronyms.

    When appropriate, provide the full financial term followed by its
    acronym.

    Examples:

    VaR -> Value at Risk (VaR)
    ROA -> Return on Assets (ROA)
    HSA -> Health Savings Account (HSA)

    Preserve the original meaning and financial figures.

    Produce the adjusted transcript followed by a list of terminology
    that was changed.
    """

    prompt_input = system_prompt + "\n\nTranscript:\n" + ascii_transcript

    messages = [
        {
            "role": "user",
            "content": prompt_input
        }
    ]

    chat_parameters = TextChatParameters(
        temperature=0.2,
        top_p=0.6
    )

    granite_model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=PROJECT_ID,
        params=chat_parameters
    )

    response = granite_model.chat(messages=messages)

    return response["choices"][0]["message"]["content"]


# ============================================================
# MEETING MINUTES PROMPT
# ============================================================

template = """
You are an AI meeting assistant.

Based only on the meeting transcript below, generate useful
meeting minutes.

Include:

1. A concise meeting summary
2. Key points discussed
3. Decisions made
4. Actionable tasks
5. Assignees and deadlines when mentioned
6. Relevant notes or context

Do not invent information that does not appear in the transcript.

Meeting Transcript:

{context}

Generate the meeting minutes now.
"""

prompt = ChatPromptTemplate.from_template(template)


# ============================================================
# LANGCHAIN PIPELINE
# ============================================================

chain = (
    {"context": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# AUDIO PROCESSING PIPELINE
# ============================================================

def transcript_audio(audio_file):
    """
    Convert meeting audio into structured meeting minutes.
    """

    if audio_file is None:
        return "Please upload an audio file.", None

    # Step 1: Speech-to-text
    raw_transcript = speech_to_text(
        audio_file,
        batch_size=8
    )["text"]

    # Step 2: Clean transcript
    ascii_transcript = remove_non_ascii(raw_transcript)

    # Step 3: Normalize financial terminology
    adjusted_transcript = product_assistant(
        ascii_transcript
    )

    # Step 4: Generate structured meeting minutes
    result = chain.invoke(
        {"context": adjusted_transcript}
    )

    # Step 5: Create downloadable output
    output_file = "meeting_minutes_and_tasks.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(result)

    return result, output_file


# ============================================================
# GRADIO INTERFACE
# ============================================================

audio_input = gr.Audio(
    sources="upload",
    type="filepath",
    label="Upload Meeting Audio"
)

output_text = gr.Textbox(
    label="Meeting Minutes and Tasks",
    lines=20
)

download_file = gr.File(
    label="Download Meeting Minutes"
)

app = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=[
        output_text,
        download_file
    ],
    title="AI Meeting Assistant",
    description=(
        "Upload a meeting audio file. The application uses "
        "Whisper for speech-to-text and IBM Granite through "
        "watsonx.ai to normalize terminology and generate "
        "structured meeting minutes and action items."
    )
)


if __name__ == "__main__":
    app.launch()
