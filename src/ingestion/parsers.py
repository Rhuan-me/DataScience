from pathlib import Path
from typing import Dict, Any, Union
from pypdf import PdfReader


def read_txt(file_path: Path) -> str:
    """Lê um ficheiro de texto simples utilizando codificação UTF-8."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(file_path: Path) -> str:
    """Extrai texto de todas as páginas de um ficheiro PDF."""
    reader = PdfReader(str(file_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    return "\n".join(pages_text)


def load_document(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Identifica a extensão do ficheiro (.txt ou .pdf), extrai o texto
    e retorna o conteúdo estruturado com metadados básicos.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {path}")

    ext = path.suffix.lower()
    if ext == ".txt":
        content = read_txt(path)
    elif ext == ".pdf":
        content = read_pdf(path)
    else:
        raise ValueError(f"Formato não suportado: {ext}. Utilize .txt ou .pdf.")

    return {"filename": path.name, "extension": ext, "content": content}
