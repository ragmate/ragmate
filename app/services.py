import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pathspec
import torch
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pygments.lexers import get_lexer_for_filename
from pygments.util import ClassNotFound
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers import Observer

from .config import get_settings
from .shared_state import CheckoutEventState

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_path_hash(filepath: str) -> str:
    return hashlib.sha256(filepath.encode()).hexdigest()


class DirectoryInspectorService:

    @staticmethod
    def is_text_file(filename: str) -> bool:
        return Path(filename).suffix.lower() in get_settings().TEXT_FILE_EXTENSIONS

    @staticmethod
    def load_patterns(project_path: str) -> pathspec.PathSpec:
        patterns = []
        ignore_files = [".gitignore", ".git/info/exclude", ".aiignore"]

        for ignore_file in ignore_files:
            root_path = os.path.join(project_path, ignore_file)
            if os.path.exists(root_path):
                with open(root_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)

        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

        for dirpath, dirnames, filenames in os.walk(project_path):
            rel_dir = os.path.relpath(dirpath, project_path)
            rel_dir = "" if rel_dir == "." else rel_dir

            for ignore_file in ignore_files:
                if ignore_file in filenames:
                    ignore_path = os.path.join(dirpath, ignore_file)
                    try:
                        with open(ignore_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line or line.startswith("#"):
                                    continue
                                pattern = os.path.join(rel_dir, line) if rel_dir else line
                                patterns.append(pattern)
                        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
                    except OSError as e:
                        logger.warning(f"Failed to read {ignore_path}: {e}")

            original_dirnames = list(dirnames)
            dirnames[:] = [d for d in dirnames if not spec.match_file(os.path.join(rel_dir, d) + "/")]
            ignored = set(original_dirnames) - set(dirnames)
            if ignored:
                logger.debug(f"Ignored subdirectories in {dirpath}: {ignored}")

        logger.info(f"Loaded {len(patterns)} patterns.")
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def list_files(self, project_path: str) -> list[str]:
        logger.info(f"Listing files in {project_path}...")

        project_path = os.path.abspath(project_path)
        spec = self.load_patterns(project_path=project_path)
        result_files = []
        for dirpath, _, filenames in os.walk(project_path):
            for file in filenames:
                if not self.is_text_file(file):
                    continue

                abs_path = os.path.join(dirpath, file)
                rel_path = os.path.relpath(abs_path, project_path)
                if not spec.match_file(rel_path):
                    result_files.append(abs_path)

        logger.info(f"Found {len(result_files)} files.")
        return result_files


class VectorStoreService:
    vector_store: Chroma | None = None

    settings = get_settings()
    _directory_inspector_service = DirectoryInspectorService()

    def _get_embeddings(self) -> Embeddings:
        api_key = self.settings.EMBEDDING_API_KEY or self.settings.LLM_API_KEY

        embeddings: Embeddings
        match self.settings.EMBEDDING_PROVIDER:
            case "openai":
                embeddings = OpenAIEmbeddings(model=self.settings.LLM_EMBEDDING_MODEL, api_key=api_key)
            case "huggingface":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.settings.LLM_EMBEDDING_MODEL,
                    cache_folder=self.settings.HUGGINGFACE_MODEL_PATH,
                    model_kwargs={"device": device},
                )
            case _:
                raise Exception(f"Unknown embedding provider: {self.settings.EMBEDDING_PROVIDER}")

        return embeddings

    def init(self, collection_name) -> bool:
        logger.info(f"Initializing vector store for collection={collection_name}...")
        persistent_client = PersistentClient(path=self.settings.CHROMA_PERSIST_PATH)
        is_created = False

        try:
            persistent_client.get_collection(collection_name)
        except NotFoundError:
            persistent_client.create_collection(collection_name)
            is_created = True

        embeddings = self._get_embeddings()

        self.vector_store = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        return is_created

    def get_retriever(self) -> "VectorStoreRetriever":
        logger.info("Getting retriever...")

        if self.vector_store is None:
            raise Exception("Vector store is not initialized.")
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

    def add_documents(self, files_list: list[str]) -> None:
        logger.info("Adding documents...")

        if self.vector_store is None:
            raise Exception("Vector store is not initialized.")

        for f in files_list:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    page_content = file.read()
                    if not page_content:
                        continue

                    doc_id = get_path_hash(filepath=f)
                    doc = Document(page_content=page_content, metadata={"source": f, "doc_id": doc_id})

                    try:
                        lexer = get_lexer_for_filename(f)
                        child_splitter = RecursiveCharacterTextSplitter.from_language(language=lexer.name.lower())
                    except ClassNotFound:
                        child_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

                    fs = LocalFileStore(self.settings.PARENT_DOCSTORE_PATH)
                    store = create_kv_docstore(fs)
                    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

                    retriever = ParentDocumentRetriever(
                        vectorstore=self.vector_store,
                        docstore=store,
                        child_splitter=child_splitter,
                        parent_splitter=parent_splitter,
                        id_key="doc_id",
                    )
                    retriever.add_documents(documents=[doc])
            except FileNotFoundError:
                logger.warning(f"File not found: {f}")

        logger.info("Documents added.")

    def remove_documents(self, ids: list[str]) -> None:
        logger.info("Removing documents...")

        if self.vector_store is None:
            raise Exception("Vector store is not initialized.")

        result = self.vector_store.get(where={"doc_id": {"$in": ids}})  # type: ignore[dict-item]
        if result["ids"]:
            self.vector_store.delete(ids=result["ids"])
            logger.info(f"Deleted {len(result['ids'])} documents.")
        logger.info("Documents removed.")


class DirectoryMonitorEventHandler(PatternMatchingEventHandler):
    changed_files: list[str] = []

    _directory_inspector_service = DirectoryInspectorService()

    def __init__(self, vector_store: VectorStoreService) -> None:
        load_patterns = self._directory_inspector_service.load_patterns(project_path=get_settings().PROJECT_PATH)
        ignore_patterns = [x.pattern for x in load_patterns.patterns]  # type: ignore[attr-defined]
        asterisked_patterns = [f"*{x}" for x in get_settings().TEXT_FILE_EXTENSIONS]

        super().__init__(patterns=asterisked_patterns, ignore_patterns=ignore_patterns, ignore_directories=True)

        self.change_counters = 0
        self.vector_store = vector_store

    def on_modified(self, event: FileSystemEvent) -> None:
        event_src_path = str(event.src_path)

        logger.info(f"Increase change counter for file: {event_src_path}")
        self.change_counters += 1

        if event_src_path not in self.changed_files:
            self.changed_files.append(event_src_path)

        if self.change_counters > 0 and self.change_counters % get_settings().REINDEX_AFTER_N_CHANGES == 0:
            logger.info(f"Reindexing files: {self.changed_files}")

            file_ids = [get_path_hash(x) for x in self.changed_files]
            self.vector_store.remove_documents(ids=file_ids)
            self.vector_store.add_documents(files_list=self.changed_files)

            self.changed_files = []

        return None

    def on_deleted(self, event: FileSystemEvent) -> None:
        event_src_path = str(event.src_path)
        logger.info(f"Deleted file: {event_src_path}")

        self.vector_store.remove_documents(ids=[get_path_hash(event_src_path)])
        if event_src_path in self.changed_files:
            self.changed_files.remove(event_src_path)

        return None

    def on_created(self, event: FileSystemEvent) -> None:
        event_src_path = str(event.src_path)
        logger.info(f"Created file: {event_src_path}")

        self.vector_store.add_documents(files_list=[event_src_path])
        if event_src_path not in self.changed_files:
            self.changed_files.append(event_src_path)

        return None

    def on_moved(self, event: FileSystemEvent) -> None:
        event_src_path = str(event.src_path)
        event_dest_path = str(event.dest_path)
        logger.info(f"Moved file: {event_src_path} to {event_dest_path}")

        self.vector_store.remove_documents(ids=[get_path_hash(event_src_path)])
        self.vector_store.add_documents(files_list=[event_dest_path])

        if event_src_path in self.changed_files:
            self.changed_files.remove(event_src_path)
        if event_dest_path not in self.changed_files:
            self.changed_files.append(event_dest_path)

        return None


class DirectoryMonitorService:
    _directory_inspector_service = DirectoryInspectorService()

    def __init__(self, vector_store: VectorStoreService, event_state: CheckoutEventState) -> None:
        self.vector_store = vector_store
        self.event_state = event_state

    def watch_directory(self) -> None:
        project_path = get_settings().PROJECT_PATH

        is_created = self.vector_store.init(collection_name=self.event_state.new_head)

        if is_created:
            logger.info("No indexed files found. Loading and indexing files...")
            files = self._directory_inspector_service.list_files(project_path=project_path)
            self.vector_store.add_documents(files_list=files)

        spec = self._directory_inspector_service.load_patterns(project_path=project_path)
        observer = Observer()
        handler = DirectoryMonitorEventHandler(vector_store=self.vector_store)

        for dirpath, dirnames, _ in os.walk(project_path):
            rel_dir = os.path.relpath(dirpath, project_path)
            rel_dir = "" if rel_dir == "." else rel_dir

            if spec.match_file(f"{rel_dir}/"):
                dirnames[:] = []
                continue

            observer.schedule(handler, dirpath, recursive=False)

        observer.start()
        logger.info("Directory monitor started.")

        try:
            while True:
                new_head = self.event_state.wait_for_new_head(timeout=1)
                if new_head:
                    logger.info(f"Received new_head in watcher: {new_head}")
                    is_created = self.vector_store.init(collection_name=new_head)

                    if is_created:
                        logger.info(f"Collection={new_head}. No indexed files found. Loading and indexing files...")
                        files = self._directory_inspector_service.list_files(project_path=project_path)
                        self.vector_store.add_documents(files_list=files)
                    else:
                        logger.info(f"Collection={new_head}. Indexed files found. Skipping...")

                time.sleep(0.1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


class CheckoutStorageService:
    settings = get_settings()

    def save_new_head(self, new_head: str) -> None:
        try:
            os.makedirs(os.path.dirname(self.settings.NEW_HEAD_FILE_PATH), exist_ok=True)
            with open(self.settings.NEW_HEAD_FILE_PATH, "w", encoding="utf-8") as f:
                f.write(new_head)
        except Exception as e:
            logger.error(f"Failed to save new_head to file: {e}")

    def load_head(self) -> str:
        try:
            if os.path.exists(self.settings.NEW_HEAD_FILE_PATH):
                with open(self.settings.NEW_HEAD_FILE_PATH, "r", encoding="utf-8") as f:
                    return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to load new_head from file: {e}")
        return self.settings.DEFAULT_COLLECTION_NAME
