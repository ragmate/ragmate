import threading


class CheckoutEventState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.new_head: str | None = None
        self._event = threading.Event()

    def set_new_head(self, new_head: str) -> None:
        with self._lock:
            self.new_head = new_head
            self._event.set()

    def wait_for_new_head(self, timeout=None) -> str | None:
        is_set = self._event.wait(timeout=timeout)
        if not is_set:
            return None
        with self._lock:
            new_head = self.new_head
            self._event.clear()
            return new_head
