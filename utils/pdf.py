from pdfminer.high_level import extract_text

def convert_pdf_to_text(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as a PDF.
    """
    try:
        text = extract_text(pdf_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {pdf_path}")
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")

    return text or ""
