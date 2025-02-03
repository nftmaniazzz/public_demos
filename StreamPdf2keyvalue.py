from typing import Optional
import streamlit as st
from pydantic import BaseModel, Field, create_model
import openai
import json
import re
import pandas as pd
import pymupdf4llm
import os

# Define the Pydantic model for the extracted data
class ExtractedData(BaseModel):
    buyer_name: str = Field(..., description="Name of the buyer on contract")
    seller_name: str = Field(..., description="Name of the seller on contract")
    buyer_address: str = Field(..., description="Address of the buyer on contract")
    seller_address: str = Field(..., description="Address of the seller on contract")
    product_name: str = Field(..., description="Name of the product with any prefixes or suffixes")
    quantity: str = Field(..., description="Description of the quantity of the product")
    quality: str = Field(..., description="Description of the quality of the product in scientific terms")
    price_or_pricing: str = Field(..., description="Description of how the price is calculated")
    payment_terms: str = Field(..., description="Description of payment terms")


# Function to validate and normalize the schema
def validate_and_normalize_schema(schema: dict) -> dict:
    # Ensure the schema has a top-level "type": "object"
    if "type" not in schema or schema["type"] != "object":
        schema = {
            "type": "object",
            "properties": schema,
            "required": list(schema.keys())  # Assume all fields are required by default
        }
    return schema
# Function to split the Markdown content into manageable chunks
def split_markdown_content(markdown_content: str, max_chunk_size: int = 100000) -> list:
    """
    Splits the Markdown content into chunks based on logical boundaries (e.g., headers).
    Ensures each chunk is less than `max_chunk_size` tokens.
    """
    chunks = []
    current_chunk = ""
    lines = markdown_content.split("\n")

    for line in lines:
        # Check if adding the next line exceeds the token limit
        if len(current_chunk) + len(line) > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        # Add the line to the current chunk
        current_chunk += line + "\n"

    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# Function to call the LLM and extract data using structured output
def extract_entities_from_markdown(markdown_content: str, schema: dict) -> Optional[BaseModel]:
    schema = validate_and_normalize_schema(schema)

    # Define the prompt template
    prompt = f"""
    You are an expert in extracting structured information from contracts. 
    Given the following Markdown content, extract the required entities and paragraphs:

    {markdown_content}

    The extracted data should include the following fields:
    {json.dumps(schema, indent=2)}

    Return the extracted data in JSON format.
    """

    # Define the structured output schema
    tool_schema = {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Extracts structured data from contract text.",
            "parameters": schema
        }
    }

    try:
        # Call the OpenAI API with structured output
        response = openai.chat.completions.create(
            model="gpt-4o",  # Replace with your desired model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            tools=[tool_schema],  # Use the 'tools' parameter for structured output
            tool_choice={"type": "function", "function": {"name": "extract_entities"}},  # Force the LLM to use the schema
            max_tokens=500,
            temperature=0.0  # Use low temperature for deterministic output
        )

        # Parse the tool call response
        tool_response = response.choices[0].message.tool_calls[0].function.arguments
        extracted_data = json.loads(tool_response)

        # Dynamically create a Pydantic model based on the schema
        DynamicModel = create_model(
            "DynamicModel",
            **{
                field: (str, ...) for field in schema["properties"].keys()
            }
        )

        # Validate the extracted data using the dynamically created model
        return DynamicModel(**extracted_data)
    except Exception as e:
        st.error(f"Error during LLM processing: {e}")
        return None

# Function to highlight text in Markdown
def highlight_text_in_markdown(markdown_content: str, extracted_data: ExtractedData) -> str:
    highlighted_content = markdown_content
    for field, value in extracted_data.model_dump().items():
        if value:  # Only highlight non-empty values
            # Escape special characters in the value for regex matching
            escaped_value = re.escape(value)
            # Wrap the matched text in <mark> tags for highlighting
            highlighted_content = re.sub(
                rf"({escaped_value})", r"<mark>\1</mark>", highlighted_content, flags=re.IGNORECASE
            )
    return highlighted_content


if __name__ == "__main__":
    st.title("Contract Entity Extraction Viewer")

    # Sidebar to upload Markdown file
    st.sidebar.header("Upload Files")
    uploaded_file = st.sidebar.file_uploader("Upload Contract File (Markdown or PDF)", type=["md", "pdf"])
    requirements_file = st.sidebar.file_uploader("Upload Extraction Requirements (JSON)", type=["json"])

    # Input for OpenAI API key
    env_openai_key = os.environ["OPENAI_API_KEY"]
    if not env_openai_key:
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    else:
        openai_api_key = env_openai_key
        
    if openai_api_key:
        openai.api_key = openai_api_key

    # Process uploaded file
    if uploaded_file and openai_api_key:
        # Split layout into two columns
        col1, col2 = st.columns([1, 2])
        if uploaded_file.type == "application/pdf":
            try:
                # Save the uploaded PDF to a temporary file
                temp_pdf_path = "temp.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Convert the PDF to Markdown using docling
                markdown_content = pymupdf4llm.to_markdown(temp_pdf_path)

                st.success("PDF successfully converted to Markdown.")
            except Exception as e:
                st.error(f"Error converting PDF to Markdown: {e}")
                markdown_content = None
        else:
            # Read the uploaded Markdown file
            markdown_content = uploaded_file.read().decode("utf-8")
# Load extraction requirements from uploaded JSON file or use default
        if requirements_file:
            try:
                extraction_requirements = json.load(requirements_file)
            except Exception as e:
                st.error(f"Error loading extraction requirements: {e}")
                extraction_requirements = None
        else:
            extraction_requirements = ExtractedData.model_json_schema()
            # Split the Markdown content into chunks
        chunks = split_markdown_content(markdown_content)
        # Split layout into two columns
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Extracted Data")
            if st.button("Extract Entities"):
                with st.spinner("Processing..."):
                        # Process each chunk and aggregate results
                    aggregated_results = {}
                    for i, chunk in enumerate(chunks):
                        with st.spinner(f"Processing chunk {i + 1}/{len(chunks)}..."):
                            extracted_data = extract_entities_from_markdown(chunk, extraction_requirements)

                        if extracted_data:
                            extracted_dict = extracted_data.model_dump()
                            for key, value in extracted_dict.items():
                                if value:  # Only store non-empty values
                                    if key not in aggregated_results:
                                        aggregated_results[key] = []
                                    aggregated_results[key].append(value)

                    # Display the aggregated results
                    st.subheader("Aggregated Extracted Data")
                    st.json(aggregated_results)
                    #extracted_data = extract_entities_from_markdown(markdown_content, extraction_requirements)

                if extracted_data:
                    st.success("Entities extracted successfully!")
                    extracted_dict = extracted_data.model_dump()
                    st.json(extracted_dict)

                    # Store extracted data in session state for highlighting and export
                    st.session_state.extracted_data = extracted_dict

                    # Export as Excel
                    df = pd.DataFrame([extracted_dict]).T.reset_index()
                    df.columns = ['Field', 'Value'] 
                    excel_buffer = df.to_excel("extracted_data.xlsx", index=False, engine="openpyxl")
                    st.download_button(
                        label="Download Extracted Data as Excel",
                        data=open("extracted_data.xlsx", "rb").read(),
                        file_name="extracted_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        with col2:
            st.subheader("Uploaded Markdown Content")
            if "highlighted_content" in st.session_state:
                # Display the highlighted Markdown content
                st.markdown(st.session_state.highlighted_content, unsafe_allow_html=True)
            else:
                # Display the original Markdown content
                st.code(markdown_content, language="markdown")

            # Highlight button
            if "extracted_data" in st.session_state:
                if st.button("Highlight Extracted Text"):
                    # Highlight the extracted text in the Markdown content
                    highlighted_content = highlight_text_in_markdown(markdown_content, st.session_state.extracted_data)
                    st.session_state.highlighted_content = highlighted_content
                    st.experimental_rerun()  # Refresh the page to show the highlighted content
    else:
        st.warning("Please upload a Markdown file and provide your OpenAI API key.")
