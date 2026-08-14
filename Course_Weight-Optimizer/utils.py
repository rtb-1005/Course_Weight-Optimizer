# utils.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    conflicts: List[tuple[str, str]] = field(default_factory=list)


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

    conflicts: List[tuple[str, str]] = []
    seen_conflicts = set()
    for index, raw_pair in enumerate(data.get("conflicts", [])):
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise ValueError(f"conflicts[{index}] must be a two-item list of course IDs.")
        left, right = (str(raw_pair[0]), str(raw_pair[1]))
        if left == right:
            raise ValueError(f"conflicts[{index}] cannot contain the same course twice.")
        if left not in courses or right not in courses:
            raise ValueError(
                f"conflicts[{index}] references an unknown course: {left}, {right}."
            )
        pair = tuple(sorted((left, right)))
        if pair not in seen_conflicts:
            conflicts.append(pair)
            seen_conflicts.add(pair)

    return GlobalState(grade_size=grade_size, courses=courses, conflicts=conflicts)
