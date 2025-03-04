import marimo

__generated_with = "0.11.13"
app = marimo.App(auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Multimodal Transcription of Medieval Handwriting: OpenAI vs Google Gemini""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Model Selection
        Choose which AI model provider you want to use for transcription.
        """
    )
    return


@app.cell
def _(mo):
    # Model selection
    model_provider = mo.ui.dropdown(
        options=["OpenAI", "Google Gemini"],
        value="OpenAI",
        label="Choose AI Model Provider"
    )
    
    model_provider
    return (model_provider,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## API Keys
        Enter your API keys for the selected model provider.
        """
    )
    return


@app.cell
def _(model_provider, mo):
    OPENAI_API_KEY = mo.ui.text(placeholder="OPENAI API KEY ...", kind="password")
    GEMINI_API_KEY = mo.ui.text(placeholder="GEMINI API KEY ...", kind="password")
    
    # Display the appropriate API key input based on selected provider
    if model_provider.value == "OpenAI":
        mo.output.replace(OPENAI_API_KEY)
    else:
        mo.output.replace(GEMINI_API_KEY)
    
    return (GEMINI_API_KEY, OPENAI_API_KEY)


@app.cell
def _(GEMINI_API_KEY, OPENAI_API_KEY, model_provider):
    import os
    from openai import OpenAI
    import google.generativeai as genai
    from pydantic import BaseModel, Field
    import json
    
    client = None
    
    # Initialize the appropriate client based on model provider
    if model_provider.value == "OpenAI" and OPENAI_API_KEY.value:
        client = OpenAI(api_key=OPENAI_API_KEY.value)
    elif model_provider.value == "Google Gemini" and GEMINI_API_KEY.value:
        # Configure with new client-based API for better structured output support
        genai.configure(api_key=GEMINI_API_KEY.value)
        client = genai
    
    return (BaseModel, Field, OpenAI, client, genai, json, os)


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
    from PIL import Image
    import io

    def encode_image_to_base64(image_file):
        """Utility function to encode image file to base64."""
        if image_file:
            return base64.b64encode(image_file).decode("utf-8")
        return None
    
    def get_image_for_gemini(image_file):
        """Utility function to get image for Gemini API."""
        if image_file:
            return Image.open(io.BytesIO(image_file))
        return None

    # Handle image
    if file_upload.value:
        base64_image = encode_image_to_base64(file_upload.value[0].contents)
        pil_image = get_image_for_gemini(file_upload.value[0].contents)
    else:
        base64_image = None
        pil_image = None
    
    return base64, base64_image, encode_image_to_base64, get_image_for_gemini, io, pil_image, Image


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
    class TranscriptionPage(BaseModel):
        text: str = Field(..., description="The transcribed text of the page in Markdown.")

    class TranscriptionError(BaseModel):
        text: str = Field(..., description="Error message when transcription fails.")
        is_error: bool = Field(default=True, description="Flag indicating this is an error.")

    def get_transcription(client, model_provider, system_prompt, content, response_format=None):
        try:
            if model_provider == "OpenAI":
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ],
                    response_format=response_format,
                )
                return completion.choices[0].message.parsed
            
            elif model_provider == "Google Gemini":
                # Configure Gemini model with structured output
                generation_config = {
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "response_mime_type": "application/json",
                    "response_schema": TranscriptionPage,
                }
                
                # Get Gemini model based on whether we're doing multimodal or text-only
                model = client.GenerativeModel(
                    model_name="gemini-1.5-pro",
                    generation_config=generation_config,
                )
                
                # Prepare prompt with system prompt included
                all_content = [
                    {"role": "user", "parts": [system_prompt]},
                    {"role": "user", "parts": content}
                ]
                
                # Generate content with structured output
                response = model.generate_content(all_content)
                
                # Parse the response and return as a TranscriptionPage object
                try:
                    # Check if response is already parsed
                    if hasattr(response, 'parsed') and response.parsed:
                        return response.parsed
                    # Otherwise try to parse the text
                    elif hasattr(response, 'text'):
                        # For older versions of the Gemini SDK
                        json_data = json.loads(response.text)
                        return TranscriptionPage(**json_data)
                    else:
                        return TranscriptionError(text="Failed to parse Gemini response")
                except Exception as parse_error:
                    return TranscriptionError(text=f"Error parsing response: {str(parse_error)}")
                
        except Exception as e:
            # Return an object with a text attribute for consistent interface
            return TranscriptionError(text=f"Error: {str(e)}")
    
    return (
        TranscriptionError,
        TranscriptionPage,
        get_transcription,
        json,
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
def _(
    TranscriptionPage,
    base64_image,
    client,
    file_upload,
    get_transcription,
    json,
    model_provider,
    mo,
    pil_image,
    prompt_text,
    run_button,
    zero_shot_set,
):
    # run Zero-Shot Transcription
    zero_shot_result = None
    zero_shot_status = None
    zero_shot_api_messages = None
    zero_shot_api_content = None

    if run_button.value:
        try:
            if file_upload.value and client:
                # Start the loading spinner
                zero_shot_model_name = "OpenAI GPT-4o" if model_provider.value == "OpenAI" else "Google Gemini 1.5 Pro"
                zero_shot_status = mo.status.spinner(
                    title=f"Transcribing Image with {zero_shot_model_name}",
                    subtitle=f"Sending request to {model_provider.value}...",
                )

                # Run the transcription
                with zero_shot_status:
                    if model_provider.value == "OpenAI":
                        zero_shot_api_messages = [
                            {"type": "text", "text": "Please transcribe this handwritten text:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                        
                        zero_shot_result = get_transcription(
                            client=client,
                            model_provider=model_provider.value,
                            system_prompt=prompt_text.value,
                            content=zero_shot_api_messages,
                            response_format=TranscriptionPage
                        )
                    else:  # Google Gemini
                        zero_shot_api_content = [
                            "Please transcribe this handwritten text:",
                            pil_image
                        ]
                        
                        zero_shot_result = get_transcription(
                            client=client,
                            model_provider=model_provider.value,
                            system_prompt=prompt_text.value,
                            content=zero_shot_api_content
                        )

                # Check if result is an error
                if hasattr(zero_shot_result, 'is_error') and zero_shot_result.is_error:
                    zero_shot_set(f"**Error**: {zero_shot_result.text}")
                else:
                    # Create the output using mo.hstack and convert to HTML string
                    zero_shot_output_content = mo.hstack(
                        items=[
                            mo.image(file_upload.value[0].contents), 
                            mo.md(f"### Transcription ({model_provider.value}): \n {zero_shot_result.text}")
                        ],
                        widths="equal",
                        gap=1,
                        align="start",
                        justify="center"
                    )
                    zero_shot_set(zero_shot_output_content._repr_html_())
            else:
                if not file_upload.value:
                    zero_shot_set("**Please upload an image file first!**")
                elif not client:
                    zero_shot_set(f"**Please enter a valid {model_provider.value} API key!**")
        except Exception as e:
            zero_shot_set(f"**Error**: {e}")
    return (
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
def _(encode_image_to_base64, get_image_for_gemini, mo, one_shot_file_upload):
    if one_shot_file_upload.value:
        one_shot_base64_image = encode_image_to_base64(one_shot_file_upload.value[0].contents)
        one_shot_pil_image = get_image_for_gemini(one_shot_file_upload.value[0].contents)
        mo.output.replace(mo.md("**Target image uploaded successfully!**"))
    else:
        one_shot_base64_image = None
        one_shot_pil_image = None
    return (one_shot_base64_image, one_shot_pil_image)


@app.cell
def _(encode_image_to_base64, example_file_upload, get_image_for_gemini):
    if example_file_upload.value:
        example_base64_image = encode_image_to_base64(example_file_upload.value[0].contents)
        example_pil_image = get_image_for_gemini(example_file_upload.value[0].contents)
    else:
        example_base64_image = None
        example_pil_image = None
    return (example_base64_image, example_pil_image)


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
    example_pil_image,
    example_transcription,
    get_transcription,
    json,
    mo,
    model_provider,
    one_shot_base64_image,
    one_shot_file_upload,
    one_shot_pil_image,
    one_shot_prompt_text,
    one_shot_run_button,
    one_shot_set,
):
    # run One-Shot Transcription
    one_shot_result = None
    one_shot_status = None
    one_shot_api_messages = None
    one_shot_api_content = None

    if one_shot_run_button.value:
        try:
            if one_shot_file_upload.value and example_file_upload.value and example_transcription.value and client:
                # Start the loading spinner
                one_shot_model_name = "OpenAI GPT-4o" if model_provider.value == "OpenAI" else "Google Gemini 1.5 Pro"
                one_shot_status = mo.status.spinner(
                    title=f"Transcribing Image with One-Shot Learning using {one_shot_model_name}",
                    subtitle=f"Sending request to {model_provider.value}...",
                )

                # Prepare the messages with example and target
                with one_shot_status:
                    if model_provider.value == "OpenAI":
                        one_shot_api_messages = [
                            {"type": "text", "text": "Here is an example of handwritten text:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_base64_image}"}},
                            {"type": "text", "text": f"The correct transcription is:\n{example_transcription.value}\n\nNow please transcribe this new text:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{one_shot_base64_image}"}}
                        ]

                        one_shot_result = get_transcription(
                            client=client,
                            model_provider=model_provider.value,
                            system_prompt=one_shot_prompt_text.value,
                            content=one_shot_api_messages,
                            response_format=TranscriptionPage
                        )
                    else:  # Google Gemini
                        one_shot_api_content = [
                            f"Here is an example of handwritten text:",
                            example_pil_image,
                            f"The correct transcription is:\n{example_transcription.value}\n\nNow please transcribe this new text:",
                            one_shot_pil_image
                        ]
                        
                        one_shot_result = get_transcription(
                            client=client,
                            model_provider=model_provider.value,
                            system_prompt=one_shot_prompt_text.value,
                            content=one_shot_api_content
                        )

                # Check if result is an error
                if hasattr(one_shot_result, 'is_error') and one_shot_result.is_error:
                    one_shot_set(f"**Error**: {one_shot_result.text}")
                else:
                    one_shot_output_content = mo.hstack(
                        items=[
                            mo.image(one_shot_file_upload.value[0].contents), 
                            mo.md(f"### Transcription ({model_provider.value}): \n {one_shot_result.text}")
                        ],
                        widths="equal",
                        gap=1,
                        align="start",
                        justify="center"
                    )
                    one_shot_set(one_shot_output_content._repr_html_())
            else:
                one_shot_missing_items = []
                if not example_file_upload.value:
                    one_shot_missing_items.append("example image")
                if not example_transcription.value:
                    one_shot_missing_items.append("example transcription")
                if not one_shot_file_upload.value:
                    one_shot_missing_items.append("target image")
                if not client:
                    one_shot_missing_items.append(f"{model_provider.value} API key")

                one_shot_set(f"**Missing required inputs: {', '.join(one_shot_missing_items)}**")
        except Exception as e:
            one_shot_set(f"**Error**: {e}")
    return (
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
def _(encode_image_to_base64, get_image_for_gemini, htr_file_upload, mo):
    # Process the uploaded image
    if htr_file_upload.value:
        htr_base64_image = encode_image_to_base64(htr_file_upload.value[0].contents)
        htr_pil_image = get_image_for_gemini(htr_file_upload.value[0].contents)
        mo.output.replace(mo.md("**Image uploaded successfully!**"))
    else:
        htr_base64_image = None
        htr_pil_image = None
    return (htr_base64_image, htr_pil_image)


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
    htr_pil_image,
    htr_prompt_text,
    htr_run_button,
    htr_set,
    json,
    mo,
    model_provider,
):
    # Run HTR Improvement
    htr_result = None
    htr_status = None
    htr_api_messages = None
    htr_api_content = None

    if htr_run_button.value:
        try:
            if htr_file_upload.value and htr_output.value and client:
                # Start the loading spinner
                htr_model_name = "OpenAI GPT-4o" if model_provider.value == "OpenAI" else "Google Gemini 1.5 Pro"
                htr_status = mo.status.spinner(
                    title=f"Improving HTR Output with {htr_model_name}",
                    subtitle=f"Sending request to {model_provider.value}...",
                )

                # Run the transcription improvement
                with htr_status:
                    if model_provider.value == "OpenAI":
                        htr_api_messages = [
                            {"type": "text", "text": "Here is a handwritten text image:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{htr_base64_image}"}},
                            {"type": "text", "text": f"The classical HTR model produced this output:\n\n```\n{htr_output.value}\n```\n\nPlease correct any errors and provide an accurate transcription."}
                        ]

                        htr_result = get_transcription(
                            client=client,
                            model_provider=model_provider.value,
                            system_prompt=htr_prompt_text.value,
                            content=htr_api_messages,
                            response_format=TranscriptionPage
                        )
                    else:  # Google Gemini
                        htr_api_content = [
                            "Here is a handwritten text image:",
                            htr_pil_image,
                            f"The classical HTR model produced this output:\n\n```\n{htr_output.value}\n```\n\nPlease correct any errors and provide an accurate transcription."
                        ]
                        
                        htr_result = get_transcription(
                            client=client,
                            model_provider=model_provider.value,
                            system_prompt=htr_prompt_text.value,
                            content=htr_api_content
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
                        mo.md(f"### Improved Transcription ({model_provider.value}):"),
                        mo.md(f"{htr_result.text}")
                    ])
                    htr_set(htr_output_content._repr_html_())
            else:
                htr_missing_items = []
                if not htr_file_upload.value:
                    htr_missing_items.append("image")
                if not htr_output.value:
                    htr_missing_items.append("HTR output")
                if not client:
                    htr_missing_items.append(f"{model_provider.value} API key")

                htr_set(f"**Missing required inputs: {', '.join(htr_missing_items)}**")
        except Exception as e:
            htr_set(f"**Error**: {e}")
    return (
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
def _(htr_improvement, mo, one_shot, zero_shot):
    # Create the tabbed interface
    tabs = mo.ui.tabs({
        "Zero Shot": zero_shot, 
        "One Shot": one_shot,
        "HTR Improvement": htr_improvement
    })
    tabs
    return (tabs,)


@app.cell
def _(model_provider, mo):
    # Show which model provider is selected
    provider_name = model_provider.value
    model_name = "GPT-4o" if provider_name == "OpenAI" else "Gemini 1.5 Pro"
    
    mo.md(f"**Currently using: {provider_name} ({model_name})**")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run() 