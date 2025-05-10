import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pathspec
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import get_settings
from .types import VectorStoreFile

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def watched_files_store_path() -> str:
    path = get_settings().WATCHED_FILES_STORE_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


class DirectoryInspectorService:

    @staticmethod
    def is_text_file(filename: str) -> bool:
        return Path(filename).suffix.lower() in get_settings().TEXT_FILE_EXTENSIONS

    @staticmethod
    def load_patterns(project_path: str) -> pathspec.PathSpec:
        patterns = [".*", "*/.*"]
        ignore_files = [".gitignore", ".aiignore", watched_files_store_path()]

        for dirpath, _, filenames in os.walk(project_path):
            for ignore_file in ignore_files:
                if ignore_file in filenames:
                    ignore_path = os.path.join(dirpath, ignore_file)
                    with open(ignore_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        rel_base = os.path.relpath(dirpath, project_path)
                        rel_base = "" if rel_base == "." else rel_base
                        for line in lines:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue
                            pattern = os.path.join(rel_base, line) if rel_base else line
                            patterns.append(pattern)

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

    def is_file_in_watched_files(self, file_path: str, project_path: str) -> bool:
        logger.info(f"Checking if file {file_path} is in watched files...")

        spec = self.load_patterns(project_path=project_path)

        if not self.is_text_file(filename=file_path):
            return False

        rel_path = os.path.relpath(file_path, project_path)
        if not spec.match_file(rel_path):
            return True

        return False


class WatchedFilesStoreService:
    path = watched_files_store_path()

    def save_file_list(self, vector_store_files: list[VectorStoreFile]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([x.model_dump() for x in vector_store_files], f, indent=2)

    def load_file_list(self) -> list[VectorStoreFile]:
        if not os.path.exists(self.path):
            return []

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [VectorStoreFile(**item) for item in data]

    def add_files_to_list(self, vector_store_files: list[VectorStoreFile]) -> None:
        file_list = []
        vector_store_files_as_dict = [x.model_dump() for x in vector_store_files]

        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                file_list = json.load(f)

        for vector_store_file_as_dict in vector_store_files_as_dict:
            if vector_store_file_as_dict not in file_list:
                file_list.append(vector_store_file_as_dict)
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(file_list, f, indent=2)

    def remove_file_from_list(self, vector_store_file: VectorStoreFile) -> None:
        vector_store_file_as_dict = vector_store_file.model_dump()

        if not os.path.exists(self.path):
            return

        with open(self.path, "r", encoding="utf-8") as f:
            file_list = json.load(f)

        if vector_store_file_as_dict in file_list:
            file_list.remove(vector_store_file_as_dict)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(file_list, f, indent=2)

    def is_list_empty(self) -> bool:
        if not os.path.exists(self.path):
            return True
        with open(self.path, "r", encoding="utf-8") as f:
            file_list = json.load(f)
        return len(file_list) == 0

    def get_file_id_by_path(self, file_path: str) -> str | None:
        vector_store_files = self.load_file_list()
        match = next((f for f in vector_store_files if f.file_path == file_path), None)
        return match.file_id if match else None

    @staticmethod
    def update_file_id_by_path(
        vector_store_files: list[VectorStoreFile],
        updated_vector_store_files: list[VectorStoreFile],
    ) -> list[VectorStoreFile]:
        file_id_map = {f.file_path: f.file_id for f in updated_vector_store_files}

        for file in vector_store_files:
            if file.file_path in file_id_map:
                file.file_id = file_id_map[file.file_path]

        return vector_store_files


class VectorStoreService:
    vector_store: InMemoryVectorStore | None = None

    _directory_inspector_service = DirectoryInspectorService()

    @staticmethod
    def get_embedding() -> OpenAIEmbeddings:
        return OpenAIEmbeddings(model=get_settings().OPENAI_EMBEDDING_MODEL, api_key=get_settings().OPENAI_API_KEY)

    def init_vector_store(self) -> None:
        embedding = self.get_embedding()
        self.vector_store = InMemoryVectorStore(embedding=embedding)

    def load_from_storage(self) -> None:
        logger.info("Loading vector store from storage...")

        self.init_vector_store()
        embedding = self.get_embedding()

        if self.vector_store is None:
            raise Exception("Vector store is not initialized.")

        self.vector_store.load(path=get_settings().VECTOR_STORE_DUMP_PATH, embedding=embedding)

    def dump_to_storage(self) -> None:
        logger.info("Saving vector store to storage...")

        if self.vector_store is None:
            raise Exception("Vector store is not initialized.")

        self.vector_store.dump(path=get_settings().VECTOR_STORE_DUMP_PATH)

    def load_storage(self) -> None:
        logger.info("Loading vector store...")

        self.init_vector_store()
        self.load_from_storage()

        logger.info("Vector store loaded.")

    def load_storage_and_index(self, project_path: str) -> list[VectorStoreFile]:
        logger.info("Loading and indexing files...")

        self.init_vector_store()
        files = self._directory_inspector_service.list_files(project_path=project_path)

        return self.add_documents(files_list=files)

    def get_retriever(self) -> "VectorStoreRetriever":
        logger.info("Getting retriever...")

        if self.vector_store is None:
            raise Exception("Vector store is not initialized.")
        return self.vector_store.as_retriever()

    def add_documents(self, files_list: list[str]) -> list[VectorStoreFile]:
        logger.info("Adding documents...")

        if self.vector_store is None:
            raise Exception("Vector store is not initialized.")

        docs = []
        file_to_doc_id = []

        for f in files_list:
            with open(f, "r", encoding="utf-8") as file:
                doc_id = str(uuid.uuid4())
                docs.append(Document(page_content=file.read(), metadata={"source": f, "doc_id": doc_id}))
                file_to_doc_id.append(VectorStoreFile(file_id=doc_id, file_path=f))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        self.vector_store.add_documents(documents=all_splits)
        self.dump_to_storage()

        logger.info("Documents added.")
        return file_to_doc_id

    def remove_documents(self, ids: list[str]) -> None:
        logger.info("Removing documents...")

        if self.vector_store is None:
            raise Exception("Vector store is not initialized.")

        self.vector_store.delete(ids=ids)
        self.dump_to_storage()


class DirectoryMonitorEventHandler(FileSystemEventHandler):
    changed_files: list[str] = []

    _watched_files_storage = WatchedFilesStoreService()
    _directory_inspector_service = DirectoryInspectorService()

    def __init__(self, watched_files: list[str], vector_store: VectorStoreService) -> None:
        super().__init__()
        self.watched_files = watched_files
        self.change_counters = 0
        self.vector_store = vector_store

    def on_modified(self, event: FileSystemEvent) -> None:
        event_src_path = str(event.src_path)

        if event_src_path not in self.watched_files:
            return None
        else:
            logger.info(f"Increase change counter for file: {event_src_path}")
            self.change_counters += 1

        if event_src_path not in self.changed_files:
            self.changed_files.append(event_src_path)

        if self.change_counters > 0 and self.change_counters % get_settings().REINDEX_AFTER_N_CHANGES == 0:
            logger.info(f"Reindexing files: {self.changed_files}")

            file_ids = []
            for changed_file in self.changed_files:
                file_id = self._watched_files_storage.get_file_id_by_path(file_path=changed_file)
                if file_id is not None:
                    file_ids.append(file_id)
            self.vector_store.remove_documents(ids=file_ids)

            updated_vector_store_files = self._watched_files_storage.update_file_id_by_path(
                vector_store_files=self._watched_files_storage.load_file_list(),
                updated_vector_store_files=self.vector_store.add_documents(files_list=self.changed_files),
            )
            self._watched_files_storage.save_file_list(vector_store_files=updated_vector_store_files)

            self.changed_files = []

        return None

    def on_deleted(self, event: FileSystemEvent) -> None:
        event_src_path = str(event.src_path)
        logger.info(f"Deleted file: {event_src_path}")

        file_id = self._watched_files_storage.get_file_id_by_path(file_path=event_src_path)
        if file_id is None:
            return None

        vector_store_file = VectorStoreFile(file_id=file_id, file_path=event_src_path)
        self._watched_files_storage.remove_file_from_list(vector_store_file=vector_store_file)
        self.vector_store.remove_documents(ids=[file_id])

        return None

    def on_created(self, event: FileSystemEvent) -> None:
        event_src_path = str(event.src_path)

        if not self._directory_inspector_service.is_file_in_watched_files(
            file_path=event_src_path, project_path=get_settings().PROJECT_PATH
        ):
            return None

        logger.info(f"Created file: {event_src_path}")

        vector_store_files = self.vector_store.add_documents(files_list=[event_src_path])
        self._watched_files_storage.add_files_to_list(vector_store_files=vector_store_files)

        return None

    def on_moved(self, event: FileSystemEvent) -> None:
        event_src_path = str(event.src_path)
        event_dest_path = str(event.dest_path)
        logger.info(f"Moved file: {event_src_path} to {event_dest_path}")

        old_file_id = self._watched_files_storage.get_file_id_by_path(file_path=event_src_path)
        if old_file_id is None:
            return None
        old_vector_store_file = VectorStoreFile(file_id=old_file_id, file_path=event_src_path)

        self._watched_files_storage.remove_file_from_list(vector_store_file=old_vector_store_file)
        self.vector_store.remove_documents(ids=[old_file_id])

        new_vector_store_files = self.vector_store.add_documents(files_list=[event_dest_path])
        self._watched_files_storage.add_files_to_list(vector_store_files=new_vector_store_files)

        return None


class DirectoryMonitorService:
    _watched_files_storage = WatchedFilesStoreService()

    def __init__(self, vector_store: VectorStoreService) -> None:
        self.vector_store = vector_store

    def watch_directory(self) -> None:
        project_path = get_settings().PROJECT_PATH

        if self._watched_files_storage.is_list_empty():
            logger.info("No watched files found. Loading and indexing files...")

            vector_store_files = self.vector_store.load_storage_and_index(project_path=project_path)
            self._watched_files_storage.save_file_list(vector_store_files=vector_store_files)
            watched_files = [x.file_path for x in vector_store_files]
        else:
            logger.info("Watched files found. Getting list of files to watch from storage file...")

            self.vector_store.load_storage()
            watched_files = [x.file_path for x in self._watched_files_storage.load_file_list()]

        event_handler = DirectoryMonitorEventHandler(watched_files=watched_files, vector_store=self.vector_store)
        observer = Observer()
        observer.schedule(event_handler, path=project_path, recursive=True)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
