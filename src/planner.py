import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


@dataclass
class Subject:
    name:            str
    difficulty:      float
    days_left:       int
    past_score:      float
    completion_pct:  float
    target_score:    float = 75.0
    predicted_score: Optional[float] = None

    @property
    def performance_gap(self):
        return max(0.0, self.target_score - self.past_score)

    @property
    def urgency(self):
        return self.difficulty / (self.days_left + 1)

    @property
    def remaining_work(self):
        return 1.0 - (self.completion_pct / 100)

    @property
    def priority_score(self):
        gap_norm = self.performance_gap / 100
        return (self.urgency * 0.4
                + gap_norm  * 0.35
                + self.remaining_work * 0.25)


@dataclass
class DayPlan:
    date:     date
    sessions: dict = field(default_factory=dict)

    @property
    def total_hours(self):
        return sum(self.sessions.values())

    def to_dict(self):
        return {
            "date":        str(self.date),
            "sessions":    {k: round(v, 2) for k, v in self.sessions.items()},
            "total_hours": round(self.total_hours, 2),
        }


class StudyPlanner:
    def __init__(self, subjects, hours_per_day=5.0,
                 start_date=None, min_session=0.5, break_days=()):
        self.subjects      = subjects
        self.hours_per_day = hours_per_day
        self.start_date    = start_date or date.today()
        self.min_session   = min_session
        self.break_days    = set(break_days)
        self._schedule     = []

    def _priority_weights(self):
        active = [s for s in self.subjects if s.days_left > 0]
        scores = {s.name: s.priority_score for s in active}
        total  = sum(scores.values()) or 1
        return {name: score / total for name, score in scores.items()}

    def _daily_allocation(self, weights):
        allocation = {name: self.hours_per_day * w for name, w in weights.items()}
        pruned, leftover = {}, 0.0
        for name, hrs in allocation.items():
            if hrs >= self.min_session:
                pruned[name] = hrs
            else:
                leftover += hrs
        if pruned and leftover > 0:
            total_p = sum(pruned.values())
            for name in pruned:
                pruned[name] += leftover * (pruned[name] / total_p)
        return pruned

    def _apply_deadline_boost(self, allocation):
        alloc = dict(allocation)
        for s in self.subjects:
            if s.days_left <= 5 and s.name in alloc:
                alloc[s.name] *= 1.20
        total = sum(alloc.values())
        scale = self.hours_per_day / total
        return {k: v * scale for k, v in alloc.items()}

    def generate_schedule(self):
        if not self.subjects:
            return []
        max_days = max(s.days_left for s in self.subjects)
        schedule = []
        for offset in range(max_days):
            current_date = self.start_date + timedelta(days=offset)
            if current_date.weekday() in self.break_days:
                continue
            temp = []
            for s in self.subjects:
                remaining = s.days_left - offset
                if remaining > 0:
                    temp.append(Subject(
                        name=s.name, difficulty=s.difficulty,
                        days_left=remaining, past_score=s.past_score,
                        completion_pct=s.completion_pct,
                        target_score=s.target_score,
                    ))
            if not temp:
                break
            day_weights = {s.name: s.priority_score for s in temp}
            day_total   = sum(day_weights.values()) or 1
            day_weights = {k: v / day_total for k, v in day_weights.items()}
            daily_alloc = self._daily_allocation(day_weights)
            daily_alloc = self._apply_deadline_boost(daily_alloc)
            schedule.append(DayPlan(date=current_date, sessions=daily_alloc))
        self._schedule = schedule
        return schedule

    def update_progress(self, subject_name, new_completion, new_score=None):
        for s in self.subjects:
            if s.name == subject_name:
                s.completion_pct = float(np.clip(new_completion, 0, 100))
                if new_score is not None:
                    s.past_score = float(np.clip(new_score, 0, 100))
                print(f"Updated {subject_name}: completion={s.completion_pct}%")
                break
        self.generate_schedule()

    def summary(self):
        weights = self._priority_weights()
        rows = []
        for s in self.subjects:
            rows.append({
                "Subject":         s.name,
                "Difficulty":      s.difficulty,
                "Days Left":       s.days_left,
                "Past Score":      s.past_score,
                "Completion %":    s.completion_pct,
                "Priority Score":  round(s.priority_score, 4),
                "Daily Hours":     round(self.hours_per_day * weights.get(s.name, 0), 2),
            })
        return pd.DataFrame(rows).sort_values("Priority Score", ascending=False)

    def to_json(self):
        return json.dumps([d.to_dict() for d in self._schedule], indent=2)


def build_planner_from_dict(student_subjects, hours_per_day=5.0):
    subjects = [Subject(**s) for s in student_subjects]
    planner  = StudyPlanner(subjects, hours_per_day=hours_per_day)
    planner.generate_schedule()
    return planner


if __name__ == "__main__":
    sample = [
        Subject("Mathematics",     difficulty=0.85, days_left=14, past_score=55, completion_pct=40),
        Subject("Physics",         difficulty=0.80, days_left=21, past_score=62, completion_pct=55),
        Subject("English",         difficulty=0.45, days_left=30, past_score=78, completion_pct=80),
        Subject("Computer Science",difficulty=0.70, days_left=10, past_score=70, completion_pct=60),
    ]
    planner  = StudyPlanner(sample, hours_per_day=6)
    schedule = planner.generate_schedule()
    print(planner.summary().to_string(index=False))
    for day in schedule[:5]:
        print(day.date, day.sessions)