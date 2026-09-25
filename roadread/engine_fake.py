from dataclasses import dataclass

@dataclass
class Read:
    text: str
    truncated: bool
    seconds: float

class FakeEngine:
    name = "fake"

    def __init__(self, replies, seconds=0.1, truncated=None):
        self.replies, self.seconds, self.truncated = list(replies), seconds, truncated or []
        self.calls, self.sizes = 0, []

    def read(self, image, instruction, max_new_tokens):
        i = min(self.calls, len(self.replies) - 1)
        self.calls += 1
        self.sizes.append(image.size)
        return Read(self.replies[i], self.truncated[i] if i < len(self.truncated) else False, self.seconds)
