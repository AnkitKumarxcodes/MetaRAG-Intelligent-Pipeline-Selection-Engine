import os
import json
import requests
from typing import List, Optional, Union, Dict
from dataclasses import dataclass, field


# ─────────────────────────────────────────
# Document Object
# ─────────────────────────────────────────

@dataclass
class Document:
    text: str
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"Document(len={len(self.text)}, source={self.metadata.get('source')}, preview='{preview}...')"


# ─────────────────────────────────────────
# Loader
# ─────────────────────────────────────────

class DocumentLoader:
    def __init__(
        self,
        data_path: Optional[Union[str, List[str]]] = None,
        recursive: bool = True,
    ):
        if data_path is None:
            self.data_paths = []
        elif isinstance(data_path, str):
            self.data_paths = [data_path]
        else:
            self.data_paths = data_path

        self.recursive = recursive

    # ─────────────────────────────────────────
    # File Discovery
    # ─────────────────────────────────────────

    def _iter_files(self, extensions: tuple) -> List[str]:
        matched = []

        for root_path in self.data_paths:
            root_path = os.path.abspath(root_path)

            if os.path.isfile(root_path):
                if root_path.endswith(extensions):
                    matched.append(root_path)
                continue

            if not os.path.isdir(root_path):
                print(f"[Warning] Path not found: {root_path}")
                continue

            if self.recursive:
                for dirpath, _, filenames in os.walk(root_path):
                    for f in filenames:
                        if f.endswith(extensions):
                            matched.append(os.path.join(dirpath, f))
            else:
                for f in os.listdir(root_path):
                    full = os.path.join(root_path, f)
                    if os.path.isfile(full) and f.endswith(extensions):
                        matched.append(full)

        return matched

    # ─────────────────────────────────────────
    # FILE LOADERS
    # ─────────────────────────────────────────

    def load_txt(self) -> List[Document]:
        docs = []
        for path in self._iter_files((".txt",)):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(Document(
                    text=f.read(),
                    metadata={"source": path, "type": "txt"}
                ))
        return docs

    def load_pdf(self) -> List[Document]:
        from pypdf import PdfReader

        docs = []
        for path in self._iter_files((".pdf",)):
            reader = PdfReader(path)

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        text=text,
                        metadata={
                            "source": path,
                            "type": "pdf",
                            "page": i
                        }
                    ))
        return docs

    def load_html(self) -> List[Document]:
        from bs4 import BeautifulSoup

        docs = []
        for path in self._iter_files((".html", ".htm")):
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)

                if text:
                    docs.append(Document(
                        text=text,
                        metadata={"source": path, "type": "html"}
                    ))
        return docs

    def load_docx(self) -> List[Document]:
        from docx import Document as DocxDocument

        docs = []
        for path in self._iter_files((".docx",)):
            doc = DocxDocument(path)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

            if text:
                docs.append(Document(
                    text=text,
                    metadata={"source": path, "type": "docx"}
                ))
        return docs

    def load_csv(self, text_columns: Optional[List[str]] = None) -> List[Document]:
        import pandas as pd

        docs = []
        for path in self._iter_files((".csv",)):
            df = pd.read_csv(path)

            cols = text_columns if text_columns else df.select_dtypes(include="object").columns.tolist()

            for i, row in df.iterrows():
                row_text = " | ".join(
                    f"{col}: {row[col]}" for col in cols
                    if col in row and pd.notna(row[col])
                )

                if row_text.strip():
                    docs.append(Document(
                        text=row_text,
                        metadata={
                            "source": path,
                            "type": "csv",
                            "row": i
                        }
                    ))
        return docs

    def load_json(self, text_key: Optional[str] = None) -> List[Document]:
        docs = []

        for path in self._iter_files((".json",)):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if text_key and isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict) and text_key in item:
                        docs.append(Document(
                            text=str(item[text_key]),
                            metadata={
                                "source": path,
                                "type": "json",
                                "index": i
                            }
                        ))
            else:
                docs.append(Document(
                    text=json.dumps(data, indent=2),
                    metadata={"source": path, "type": "json"}
                ))

        return docs

    # ─────────────────────────────────────────
    # WEB LOADERS
    # ─────────────────────────────────────────

    def load_url(self, url: str) -> Document:
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        return Document(
            text=text,
            metadata={"source": url, "type": "web"}
        )

    def load_urls(self, urls: List[str]) -> List[Document]:
        docs = []
        for url in urls:
            try:
                doc = self.load_url(url)
                docs.append(doc)
                print(f"[✓] {url}")
            except Exception as e:
                print(f"[✗] Failed {url}: {e}")
        return docs

    def load_sitemap(self, sitemap_url: str, max_pages: int = 20) -> List[Document]:
        from bs4 import BeautifulSoup

        response = requests.get(sitemap_url, timeout=10)
        soup = BeautifulSoup(response.text, "xml")

        urls = [loc.text for loc in soup.find_all("loc")][:max_pages]
        print(f"[Sitemap] Found {len(urls)} URLs")

        return self.load_urls(urls)

    # ─────────────────────────────────────────
    # MAIN ENTRY
    # ─────────────────────────────────────────

    def load_all(
        self,
        urls: Optional[List[str]] = None,
        sitemap_url: Optional[str] = None,
        json_text_key: Optional[str] = None,
        csv_text_columns: Optional[List[str]] = None,
    ) -> List[Document]:

        docs: List[Document] = []

        if self.data_paths:
            loaders = [
                ("TXT",  self.load_txt),
                ("PDF",  self.load_pdf),
                ("HTML", self.load_html),
                ("DOCX", self.load_docx),
                ("CSV",  lambda: self.load_csv(csv_text_columns)),
                ("JSON", lambda: self.load_json(json_text_key)),
            ]

            for name, loader in loaders:
                try:
                    loaded = loader()
                    docs.extend(loaded)
                    if loaded:
                        print(f"[✓] {name}: {len(loaded)} doc(s)")
                except Exception as e:
                    print(f"[✗] {name} loader failed: {e}")

        if urls:
            docs.extend(self.load_urls(urls))

        if sitemap_url:
            docs.extend(self.load_sitemap(sitemap_url))

        print(f"\n[DocumentLoader] Total docs loaded: {len(docs)}")
        return docs