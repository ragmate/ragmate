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
        ignore_files = [".gitignore", ".aiignore"]

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

    _directory_inspector_service = DirectoryInspectorService()

    @staticmethod
    def _get_embeddings() -> Embeddings:
        settings = get_settings()
        api_key = settings.EMBEDDING_API_KEY or settings.LLM_API_KEY

        embeddings: Embeddings
        match settings.EMBEDDING_PROVIDER:
            case "openai":
                embeddings = OpenAIEmbeddings(model=settings.LLM_EMBEDDING_MODEL, api_key=api_key)
            case "huggingface":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                embeddings = HuggingFaceEmbeddings(
                    model_name=settings.LLM_EMBEDDING_MODEL,
                    cache_folder=settings.HUGGINGFACE_MODEL_PATH,
                    model_kwargs={"device": device},
                )
            case _:
                raise Exception(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}")

        return embeddings

    def init(self) -> bool:
        persistent_client = PersistentClient(path=get_settings().CHROMA_PERSIST_PATH)
        collection_name = "collection"
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
        return self.vector_store.as_retriever()

    def add_documents(self, files_list: list[str]) -> None:
        logger.info("Adding documents...")

        if self.vector_store is None:
            raise Exception("Vector store is not initialized.")

        for f in files_list:
            with open(f, "r", encoding="utf-8") as file:
                page_content = file.read()
                if not page_content:
                    return None

                doc_id = get_path_hash(filepath=f)
                doc = Document(page_content=page_content, metadata={"source": f, "doc_id": doc_id})

                try:
                    lexer = get_lexer_for_filename(f)
                    splitter = RecursiveCharacterTextSplitter.from_language(language=lexer.name.lower())
                except ClassNotFound:
                    splitter = RecursiveCharacterTextSplitter()

                all_splits = splitter.split_documents([doc])
                self.vector_store.add_documents(documents=all_splits)

        logger.info("Documents added.")

    def remove_documents(self, ids: list[str]) -> None:
        logger.info("Removing documents...")

        if self.vector_store is None:
            raise Exception("Vector store is not initialized.")

        result = self.vector_store.get(where={"doc_id": {"$in": ids}})
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

    def __init__(self, vector_store: VectorStoreService) -> None:
        self.vector_store = vector_store

    def watch_directory(self) -> None:
        project_path = get_settings().PROJECT_PATH

        is_created = self.vector_store.init()

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
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
