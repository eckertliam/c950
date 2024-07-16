from datetime import time
from dataclasses import dataclass
from typing import List, Optional
from distance import DistanceTable


@dataclass
class Truck:
    id: int
    miles: int
    """Speed in miles per hour"""
    speed: float
    max_weight: int
    current_weight: int
    """Marks the current index of the route"""
    current_location: int
    current_time: time
    route: Optional[List[int]] = None
    packages: Optional[List[int]] = None
