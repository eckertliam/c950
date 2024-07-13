from dataclasses import dataclass
from datetime import time
from typing import Optional


@dataclass
class Package:
    """Represents a package to be delivered"""
    id: int
    address: int
    weight: float
    deadline: Optional[time] = None
    delivery_time: Optional[time] = None
    delivered: bool = False
