import docx

def convert_docx_to_txt(docx_path: str) -> str:
    """
    Convert a .docx file into plain text.

    Args:
        docx_path (str): Path to the .docx file.

    Returns:
        str: Extracted text from the document, with paragraphs 
             separated by newlines.

    Raises:
        FileNotFoundError: If the given .docx file does not exist.
        ValueError: If the file is not a valid .docx document.
    """
    try:
        doc = docx.Document(docx_path)
    except Exception as e:
        raise ValueError(f"Could not read .docx file: {e}")

    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text
