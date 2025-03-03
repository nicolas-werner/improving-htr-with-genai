import marimo

__generated_with = "0.11.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Proof of Concept: Using MMLLMs to improve transcription of medieval handwriting""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Settings
        enter you Openai API Key below
        """
    )
    return


@app.cell
def _(mo):
    OPENAI_API_KEY = mo.ui.text(placeholder="OPEN AI API KEY ...", kind="password")
    OPENAI_API_KEY
    return (OPENAI_API_KEY,)


@app.cell
def _(OPENAI_API_KEY):
    import os
    from openai import OpenAI
    if OPENAI_API_KEY.value:
        client = OpenAI(api_key= OPENAI_API_KEY.value)
    return OpenAI, client, os


@app.cell
def _(mo):
    # System Prompt
    prompt_text = mo.ui.text_area(value="You are an expert for medieval handwritten middle high german. Transcribe the text in this image exactly and return it in markdown format.")
    return (prompt_text,)


@app.cell
def _(mo):
    # Image Upload
    file_upload = mo.ui.file(kind="area", filetypes=[".png", ".jpg", ".jpeg"], multiple=False, label="Upload Image")
    return (file_upload,)


@app.cell
def _(file_upload):
    import base64

    def encode_image_to_base64(image_file):
        """Utility function to encode image file to base64."""
        if image_file:
            return base64.b64encode(image_file).decode("utf-8")
        return None

    # Handle zero-shot image
    if file_upload.value:
        base64_image = encode_image_to_base64(file_upload.value[0].contents)
    else:
        base64_image = None
    return base64, base64_image, encode_image_to_base64


@app.cell
def _(mo):
    # Create state variables for outputs
    zero_shot_get, zero_shot_set = mo.state("")
    one_shot_get, one_shot_set = mo.state("")
    htr_get, htr_set = mo.state("")
    return (
        htr_get,
        htr_set,
        one_shot_get,
        one_shot_set,
        zero_shot_get,
        zero_shot_set,
    )


@app.cell
def _():
    from pydantic import BaseModel, Field

    class TranscriptionPage(BaseModel):
        text: str = Field(..., description="The transcribed text of the page in Markdown.")

    class TranscriptionError(BaseModel):
        text: str = Field(..., description="Error message when transcription fails.")
        is_error: bool = Field(default=True, description="Flag indicating this is an error.")

    def get_transcription(client, model, system_prompt, messages, response_format):

        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": messages}
                ],
                response_format=response_format,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            # Return an object with a text attribute for consistent interface
            return TranscriptionError(text=f"Error: {str(e)}")
    return (
        BaseModel,
        Field,
        TranscriptionError,
        TranscriptionPage,
        get_transcription,
    )


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run Transcription 🏃 ", kind="success")
    return (run_button,)


@app.cell
def _(file_upload, mo, prompt_text, run_button, zero_shot_get):
    zero_shot = mo.vstack([
        "System Prompt", prompt_text,
        "Image to transcribe", file_upload,
        run_button,
        mo.md(zero_shot_get())
        ])
    return (zero_shot,)


@app.cell
def _():
    # Transcription Type
    return


@app.cell
def _(htr_improvement, mo, one_shot, zero_shot):
    tabs = mo.ui.tabs({
        "Zero Shot": zero_shot, 
        "One Shot": one_shot,
        "HTR Improvement": htr_improvement
    })
    tabs
    return (tabs,)


@app.cell
def _(
    TranscriptionPage,
    base64_image,
    client,
    file_upload,
    get_transcription,
    mo,
    prompt_text,
    run_button,
    zero_shot_set,
):
    # run Zero-Shot Transcription
    zero_shot_result = None
    zero_shot_status = None

    if run_button.value:
        try:
            if file_upload.value:
                # Start the loading spinner
                zero_shot_status = mo.status.spinner(
                    title="Transcribing Image",
                    subtitle="Sending request to OpenAI...",
                )

                # Run the transcription
                with zero_shot_status:
                    zero_shot_messages = [
                        {"type": "text", "text": "Please transcribe this handwritten text:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]

                    zero_shot_result = get_transcription(
                        client=client,
                        model="gpt-4o",
                        system_prompt=prompt_text.value,
                        messages=zero_shot_messages,
                        response_format=TranscriptionPage
                    )

                # Check if result is an error
                if hasattr(zero_shot_result, 'is_error') and zero_shot_result.is_error:
                    zero_shot_set(f"**Error**: {zero_shot_result.text}")
                else:
                    # Create the output using mo.hstack and convert to HTML string
                    zero_shot_output_content = mo.hstack(
                        items=[
                            mo.image(file_upload.value[0].contents), 
                            mo.md(f"### Transcription: \n {zero_shot_result.text}")
                        ],
                        widths="equal",
                        gap=1,
                        align="start",
                        justify="center"
                    )
                    zero_shot_set(zero_shot_output_content._repr_html_())
            else:
                zero_shot_set("**Please upload an image file first!**")
        except Exception as e:
            zero_shot_set(f"**Error**: {e}")
    return (
        zero_shot_messages,
        zero_shot_output_content,
        zero_shot_result,
        zero_shot_status,
    )


@app.cell
def _(mo):
    # System Prompt for One Shot
    one_shot_prompt_text = mo.ui.text_area(value="You are an expert for medieval handwritten middle high german. Here is an example of a handwritten text and its transcription. Use this example to help you transcribe the new text. Transcribe the text in this image exactly and return it in markdown format.")
    return (one_shot_prompt_text,)


@app.cell
def _(mo):
    # Example Image Upload
    example_file_upload = mo.ui.file(kind="area", filetypes=[".png", ".jpg", ".jpeg"], multiple=False, label="Upload Example Image")
    return (example_file_upload,)


@app.cell
def _(example_file_upload, mo):
    # Example Transcription
    example_transcription = mo.ui.text_area(label="", placeholder="Enter the correct transcription for the example image...")

    if example_file_upload.value:
        mo.output.replace(mo.md("**Example image uploaded successfully!**"))
    return (example_transcription,)


@app.cell
def _(mo):
    # Target Image Upload for One Shot
    one_shot_file_upload = mo.ui.file(kind="area", filetypes=[".png", ".jpg", ".jpeg"], multiple=False, label="Upload Target Image to Transcribe")
    return (one_shot_file_upload,)


@app.cell
def _(encode_image_to_base64, mo, one_shot_file_upload):
    if one_shot_file_upload.value:
        one_shot_base64_image = encode_image_to_base64(one_shot_file_upload.value[0].contents)
        mo.output.replace(mo.md("**Target image uploaded successfully!**"))
    else:
        one_shot_base64_image = None
    return (one_shot_base64_image,)


@app.cell
def _(encode_image_to_base64, example_file_upload):
    if example_file_upload.value:
        example_base64_image = encode_image_to_base64(example_file_upload.value[0].contents)
    else:
        example_base64_image = None
    return (example_base64_image,)


@app.cell
def _(mo):
    one_shot_run_button = mo.ui.run_button(label="Run One-Shot Transcription 🏃 ", kind="success")
    return (one_shot_run_button,)


@app.cell
def _(
    TranscriptionPage,
    client,
    example_base64_image,
    example_file_upload,
    example_transcription,
    get_transcription,
    mo,
    one_shot_base64_image,
    one_shot_file_upload,
    one_shot_prompt_text,
    one_shot_run_button,
    one_shot_set,
):
    # run One-Shot Transcription
    one_shot_result = None
    one_shot_status = None

    if one_shot_run_button.value:
        try:
            if one_shot_file_upload.value and example_file_upload.value and example_transcription.value:
                # Start the loading spinner
                one_shot_status = mo.status.spinner(
                    title="Transcribing Image with One-Shot Learning",
                    subtitle="Sending request to OpenAI...",
                )

                # Prepare the messages with example and target
                with one_shot_status:
                    one_shot_messages = [
                        {"type": "text", "text": "Here is an example of handwritten text:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_base64_image}"}},
                        {"type": "text", "text": f"The correct transcription is:\n{example_transcription.value}\n\nNow please transcribe this new text:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{one_shot_base64_image}"}}
                    ]

                    one_shot_result = get_transcription(
                        client=client,
                        model="gpt-4o",
                        system_prompt=one_shot_prompt_text.value,
                        messages=one_shot_messages,
                        response_format=TranscriptionPage
                    )

                # Check if result is an error
                if hasattr(one_shot_result, 'is_error') and one_shot_result.is_error:
                    one_shot_set(f"**Error**: {one_shot_result.text}")
                else:
                    one_shot_output_content = mo.hstack(
                        items=[
                            mo.image(one_shot_file_upload.value[0].contents), 
                            mo.md(f"### Transcription: \n {one_shot_result.text}")
                        ],
                        widths="equal",
                        gap=1,
                        align="start",
                        justify="center"
                    )
                    one_shot_set(one_shot_output_content._repr_html_())
            else:
                one_shot_missing = []
                if not example_file_upload.value:
                    one_shot_missing.append("example image")
                if not example_transcription.value:
                    one_shot_missing.append("example transcription")
                if not one_shot_file_upload.value:
                    one_shot_missing.append("target image")

                one_shot_set(f"**Missing required inputs: {', '.join(one_shot_missing)}**")
        except Exception as e:
            one_shot_set(f"**Error**: {e}")
    return (
        one_shot_messages,
        one_shot_missing,
        one_shot_output_content,
        one_shot_result,
        one_shot_status,
    )


@app.cell
def _(
    example_file_upload,
    example_transcription,
    mo,
    one_shot_file_upload,
    one_shot_get,
    one_shot_prompt_text,
    one_shot_run_button,
):
    one_shot = mo.vstack([
        "System Prompt", one_shot_prompt_text,
        "Example Image", example_file_upload,
        "Example Transcription", example_transcription,
        "Target Image to Transcribe", one_shot_file_upload,
        one_shot_run_button,
        mo.md(one_shot_get())
    ])
    return (one_shot,)


@app.cell
def _(mo):
    # System Prompt for HTR Improvement
    htr_prompt_text = mo.ui.text_area(value="You are an expert for medieval handwritten middle high german. I will provide you with an image of handwritten text and the output from a classical HTR (Handwritten Text Recognition) model. This output may contain errors. Your task is to correct any errors and provide an accurate transcription. Return the corrected text in markdown format.")
    return (htr_prompt_text,)


@app.cell
def _(mo):
    # Image Upload for HTR Improvement
    htr_file_upload = mo.ui.file(kind="area", filetypes=[".png", ".jpg", ".jpeg"], multiple=False, label="Upload Image")
    return (htr_file_upload,)


@app.cell
def _(encode_image_to_base64, htr_file_upload, mo):
    # Process the uploaded image
    if htr_file_upload.value:
        htr_base64_image = encode_image_to_base64(htr_file_upload.value[0].contents)
        mo.output.replace(mo.md("**Image uploaded successfully!**"))
    else:
        htr_base64_image = None
    return (htr_base64_image,)


@app.cell
def _(mo):
    # HTR Output
    htr_output = mo.ui.text_area(label="", placeholder="Paste the output from your classical HTR model here...")
    return (htr_output,)


@app.cell
def _(mo):
    # Run button for HTR Improvement
    htr_run_button = mo.ui.run_button(label="Run HTR Improvement 🏃 ", kind="success")
    return (htr_run_button,)


@app.cell
def _(
    TranscriptionPage,
    client,
    get_transcription,
    htr_base64_image,
    htr_file_upload,
    htr_output,
    htr_prompt_text,
    htr_run_button,
    htr_set,
    mo,
):
    # Run HTR Improvement
    htr_result = None
    htr_status = None

    if htr_run_button.value:
        try:
            if htr_file_upload.value and htr_output.value:
                # Start the loading spinner
                htr_status = mo.status.spinner(
                    title="Improving HTR Output",
                    subtitle="Sending request to OpenAI...",
                )

                # Run the transcription improvement
                with htr_status:
                    htr_messages = [
                        {"type": "text", "text": "Here is a handwritten text image:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{htr_base64_image}"}},
                        {"type": "text", "text": f"The classical HTR model produced this output:\n\n```\n{htr_output.value}\n```\n\nPlease correct any errors and provide an accurate transcription."}
                    ]

                    htr_result = get_transcription(
                        client=client,
                        model="gpt-4o",
                        system_prompt=htr_prompt_text.value,
                        messages=htr_messages,
                        response_format=TranscriptionPage
                    )

                # Check if result is an error
                if hasattr(htr_result, 'is_error') and htr_result.is_error:
                    htr_set(f"**Error**: {htr_result.text}")
                else:
                    htr_output_content = mo.vstack([
                        mo.hstack(
                            items=[
                                mo.image(htr_file_upload.value[0].contents),
                                mo.vstack([
                                    mo.md("### Original HTR Output:"),
                                    mo.md(f"```\n{htr_output.value}\n```")
                                ])
                            ],
                            widths="equal",
                            gap=1,
                            align="start",
                            justify="center"
                        ),
                        mo.md("### Improved Transcription:"),
                        mo.md(f"{htr_result.text}")
                    ])
                    htr_set(htr_output_content._repr_html_())
            else:
                htr_missing = []
                if not htr_file_upload.value:
                    htr_missing.append("image")
                if not htr_output.value:
                    htr_missing.append("HTR output")

                htr_set(f"**Missing required inputs: {', '.join(htr_missing)}**")
        except Exception as e:
            htr_set(f"**Error**: {e}")
    return (
        htr_messages,
        htr_missing,
        htr_output_content,
        htr_result,
        htr_status,
    )


@app.cell
def _(
    htr_file_upload,
    htr_get,
    htr_output,
    htr_prompt_text,
    htr_run_button,
    mo,
):
    # Create the HTR improvement tab
    htr_improvement = mo.vstack([
        "System Prompt", htr_prompt_text,
        "Image", htr_file_upload,
        "Classical HTR Output", htr_output,
        htr_run_button,
        mo.md(htr_get())
    ])
    return (htr_improvement,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
