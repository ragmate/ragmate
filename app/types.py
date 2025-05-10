from pydantic import BaseModel


class VectorStoreFile(BaseModel):
    file_id: str
    file_path: str
