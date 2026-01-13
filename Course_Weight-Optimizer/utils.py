# utils.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class CoursePref:
    course_id: str
    utility: float


@dataclass
class DesiredCourses:
    preferences: List[CoursePref]


@dataclass
class CourseState:
    course_id: str
    capacity: int   # c_j
    bidders: int    # m_j (>= MIN_BID 的人数)


@dataclass
class GlobalState:
    grade_size: int                # P
    courses: Dict[str, CourseState]


def load_desired_courses(json_path: str) -> DesiredCourses:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    prefs: List[CoursePref] = []
    for item in data.get("preferences", []):
        prefs.append(CoursePref(
            course_id=str(item["course_id"]),
            utility=float(item.get("utility", 1.0)),
        ))
    return DesiredCourses(preferences=prefs)


def load_global_state(json_path: str) -> GlobalState:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if "grade_size" not in data:
        raise ValueError("global_state.json must contain 'grade_size' (P).")
    grade_size = int(data["grade_size"])
    if grade_size <= 0:
        raise ValueError("grade_size must be positive.")

    courses: Dict[str, CourseState] = {}
    for item in data.get("courses", []):
        cid = str(item["course_id"])
        courses[cid] = CourseState(
            course_id=cid,
            capacity=int(item["capacity"]),
            bidders=int(item["bidders"]),
        )

    if not courses:
        raise ValueError("global_state.json 'courses' is empty.")

    return GlobalState(grade_size=grade_size, courses=courses)
