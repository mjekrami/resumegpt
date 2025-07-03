import pdfplumber
import docx


def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])


def extract_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_bullets(text):
    return "\n".join(
        [line for line in text.splitlines() if line.strip().startswith("-")]
    )


def extract_intro(text):
    return "\n".join(
        [line for line in text.splitlines() if "objective" in line.lower()]
    )


# --- LangChain Nodes ---
def parse_resume(text):
    return {
        "bullet_points": extract_bullets(text),
        "intro_objectives": extract_intro(text),
    }
