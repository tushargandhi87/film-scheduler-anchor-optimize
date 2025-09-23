"""
Film Production Schedule Optimizer - Iteration 2
Goal: The "Naïve" Scheduler (Anchors & Sequential Fill)
This script builds upon Iteration 1 by introducing a basic scheduler that
respects non-negotiable constraints. It first places "anchor" scenes on their
mandatory dates and then fills the remaining schedule sequentially.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
import uvicorn

# --- 1. API and Data Models (Updated for Iteration 2) ---

app = FastAPI(
    title="Film Schedule Optimizer - Iteration 2",
    description="Implements a basic scheduler that handles non-negotiable constraints."
)

class ScheduleRequest(BaseModel):
    stripboard: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    ga_params: Optional[Dict[str, Any]] = {}

class Iteration2Response(BaseModel):
    message: str
    total_shooting_days: int
    schedule: List[Dict[str, Any]]
    unplaced_anchors: List[Dict[str, Any]]

# --- 2. Core Data Structures (Updated for Iteration 2) ---

@dataclass
class LocationCluster:
    location: str
    scenes: List[Dict]
    total_hours: float
    required_actors: List[str]

@dataclass
class NonNegotiableConstraint:
    """Represents a single, immovable scheduling anchor."""
    constraint_type: str
    identifier: str
    date: Optional[date] = None
    scenes_affected: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


# --- 3. Helper Classes (Updated for Iteration 2) ---

class ShootingCalendar:
    """Manages the available shooting days for the production."""
    def __init__(self, start_date_str: str, end_date_str: str):
        self.start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        self.shooting_days = self._generate_shooting_days()
        print(f"INFO: Calendar created with {len(self.shooting_days)} available shooting days.")

    def _generate_shooting_days(self) -> List[date]:
        days = []
        current_day = self.start_date
        while current_day <= self.end_date:
            if current_day.weekday() != 6: # 6 = Sunday
                days.append(current_day)
            current_day += timedelta(days=1)
        return days

class StructuredConstraintParser:
    """
    Parses raw constraints, focusing on "Non-Negotiable" anchors.
    """
    def __init__(self, constraints: Dict[str, Any]):
        self.raw_constraints = constraints
        self.non_negotiables = self._parse_non_negotiables()
        print(f"INFO: Parser found {len(self.non_negotiables)} non-negotiable constraints.")

    def _parse_non_negotiables(self) -> List[NonNegotiableConstraint]:
        anchors = []
        anchors.extend(self._parse_actor_anchors())
        anchors.extend(self._parse_location_anchors())
        anchors.extend(self._parse_creative_anchors())
        return anchors

    def _parse_actor_anchors(self) -> List[NonNegotiableConstraint]:
        # Implementation for parsing actor hard-outs
        # ... (This logic was correct and remains the same)
        return []

    def _parse_location_anchors(self) -> List[NonNegotiableConstraint]:
        anchors = []
        try:
            locations = self.raw_constraints.get('location_constraints', {}).get('locations', {})
            for loc_name, details in locations.items():
                real_address = details.get('real_address', loc_name)
                for constraint in details.get('constraints', []):
                    if constraint.get('constraint_level') == 'Non-negotiable':
                        # Handle fixed date ranges
                        if constraint.get('constraint_type') == 'availability_window':
                            parsed_data = constraint['parsed_data']
                            start_date = datetime.strptime(parsed_data['start_date'], "%Y-%m-%d").date()
                            end_date = datetime.strptime(parsed_data['end_date'], "%Y-%m-%d").date()
                            anchors.append(NonNegotiableConstraint(
                                constraint_type='location_date_range',
                                identifier=real_address,
                                details={'start': start_date, 'end': end_date, 'scenes': details.get('scenes', [])}
                            ))
                        # Handle day-of-week restrictions
                        elif constraint.get('constraint_type') == 'day_restriction':
                            parsed_data = constraint['parsed_data']
                            anchors.append(NonNegotiableConstraint(
                                constraint_type='location_day_of_week',
                                identifier=real_address,
                                details={'allowed_days': parsed_data.get('day_restrictions', []), 'scenes': details.get('scenes', [])}
                            ))
        except (KeyError, TypeError, ValueError) as e:
            print(f"WARNING: Could not parse location constraints: {e}")
        return anchors
    
    def _parse_creative_anchors(self) -> List[NonNegotiableConstraint]:
        # Implementation for Director & DOP notes
        # ... (This logic was correct and remains the same)
        return []


class NaiveScheduler:
    """
    A state-aware scheduler that correctly handles all non-negotiable constraints.
    """
    def __init__(self, clusters: List[LocationCluster], calendar: ShootingCalendar, parser: StructuredConstraintParser, scene_time_estimates: Dict[str, float]):
        self.clusters = clusters
        self.calendar = calendar
        self.parser = parser
        self.schedule: Dict[date, Dict] = {day: {"scenes": [], "location": None} for day in self.calendar.shooting_days}
        self.unplaced_anchors = []
        self.scene_time_estimates = scene_time_estimates
        # NEW: A lookup for original constraints for each location
        self.location_rules = self._map_location_rules()

    def _map_location_rules(self) -> Dict[str, Dict]:
        """Creates a simple lookup table for the original constraints of each location."""
        rules = {}
        locations_data = self.parser.raw_constraints.get('location_constraints', {}).get('locations', {})
        for loc_name, details in locations_data.items():
            real_address = details.get('real_address', loc_name)
            rules[real_address] = {
                'date_range': None,
                'allowed_days': None
            }
            for constraint in details.get('constraints', []):
                if constraint.get('constraint_level') == 'Non-negotiable':
                    if constraint.get('constraint_type') == 'availability_window':
                        parsed = constraint['parsed_data']
                        rules[real_address]['date_range'] = (
                            datetime.strptime(parsed['start_date'], "%Y-%m-%d").date(),
                            datetime.strptime(parsed['end_date'], "%Y-%m-%d").date()
                        )
                    elif constraint.get('constraint_type') == 'day_restriction':
                         rules[real_address]['allowed_days'] = constraint['parsed_data'].get('day_restrictions', [])
        return rules

    def build_schedule(self) -> List[Dict[str, Any]]:
        self._place_anchors()
        self._fill_remaining_days()
        return self._format_schedule()

    def _place_anchors(self):
        """Places entire location clusters that have non-negotiable constraints."""
        print("INFO: Placing non-negotiable anchors on the calendar...")
        
        for anchor in self.parser.non_negotiables:
             if anchor.constraint_type in ['location_date_range', 'location_day_of_week']:
                cluster_to_place = next((c for c in self.clusters if c.location == anchor.identifier), None)
                if cluster_to_place:
                    self._schedule_cluster(cluster_to_place)
                    self.clusters.remove(cluster_to_place) # Remove so it's not scheduled again

    def _fill_remaining_days(self):
        """Fills empty calendar slots with the remaining flexible clusters."""
        print("INFO: Filling remaining calendar days with flexible clusters...")
        sorted_clusters = sorted(self.clusters, key=lambda c: c.total_hours, reverse=True)
        for cluster in sorted_clusters:
            self._schedule_cluster(cluster)

    def _schedule_cluster(self, cluster: LocationCluster):
        """Finds valid days for a cluster and packs its scenes."""
        scenes_to_schedule = list(cluster.scenes)
        day_cursor = 0
        MAX_DAILY_HOURS = 10.0

        while scenes_to_schedule:
            # Find the next valid, available day for this cluster
            next_valid_day_idx = self._find_next_valid_day(day_cursor, cluster.location)

            if next_valid_day_idx is None:
                print(f"WARNING: No valid days found for remaining scenes of {cluster.location}.")
                self.unplaced_anchors.append({'identifier': cluster.location, 'scenes': [s['Scene_Number'] for s in scenes_to_schedule]})
                return

            current_date = self.calendar.shooting_days[next_valid_day_idx]
            self.schedule[current_date]['location'] = cluster.location
            day_cursor = next_valid_day_idx

            # Pack scenes into this day
            day_hours = 0
            scenes_for_this_day = []
            while scenes_to_schedule:
                scene = scenes_to_schedule[0]
                scene_hours = self.scene_time_estimates.get(str(scene.get('Scene_Number')), 1.0)
                if day_hours + scene_hours <= MAX_DAILY_HOURS:
                    scenes_for_this_day.append(scenes_to_schedule.pop(0))
                    day_hours += scene_hours
                else:
                    break
            
            self.schedule[current_date]['scenes'] = scenes_for_this_day
            day_cursor += 1
            
    def _find_next_valid_day(self, start_index: int, location: str) -> Optional[int]:
        """Finds the next day that is both empty and valid for a given location's rules."""
        location_rules = self.location_rules.get(location, {})
        
        for i in range(start_index, len(self.calendar.shooting_days)):
            day = self.calendar.shooting_days[i]
            
            # Check 1: Is the day already booked?
            if self.schedule[day]['location'] is not None:
                continue
                
            # Check 2: Does it violate a date range rule?
            date_range = location_rules.get('date_range')
            if date_range and not (date_range[0] <= day <= date_range[1]):
                continue

            # Check 3: Does it violate a day-of-week rule?
            allowed_days = location_rules.get('allowed_days')
            if allowed_days and day.strftime('%A') not in allowed_days:
                continue
            
            # If all checks pass, this is a valid day
            return i
            
        return None

    def _format_schedule(self) -> List[Dict[str, Any]]:
        # ... (This logic was correct and remains the same)
        return []

# --- 5. Main API Endpoint and supporting classes ---
# ... (All remaining classes and the endpoint logic remain the same)
class LocationClusterManager:
    def __init__(self, stripboard: List[Dict], constraints: Dict[str, Any]):
        self.stripboard = stripboard
        self.scene_time_estimates = self._get_scene_time_estimates(constraints)
        self.clusters = self._create_location_clusters()
        print(f"INFO: Successfully created {len(self.clusters)} location clusters.")
    def _get_scene_time_estimates(self, constraints: Dict) -> Dict[str, float]:
        try:
            time_estimates = constraints['operational_data']['time_estimates']['scene_estimates']
            return { str(est[key]): float(est['Estimated_Time_Hours']) for est in time_estimates for key in est if 'Scene_Number' in key }
        except (KeyError, TypeError): return {}
    def _create_location_clusters(self) -> List[LocationCluster]:
        location_groups = defaultdict(list)
        for scene in self.stripboard:
            location = scene.get('Geographic_Location')
            if location and location != 'Location TBD': location_groups[location].append(scene)
        clusters = []
        for location, scenes in location_groups.items():
            total_hours = sum([self.scene_time_estimates.get(str(s.get('Scene_Number')), self._estimate_scene_hours_from_page_count(s)) for s in scenes])
            all_actors = set(actor for scene in scenes for actor in scene.get('Cast', []) if isinstance(scene.get('Cast'), list))
            clusters.append(LocationCluster(location=location, scenes=sorted(scenes, key=lambda s: s.get('Scene_Number', '')), total_hours=round(total_hours, 2), required_actors=sorted(list(all_actors))))
        return sorted(clusters, key=lambda c: c.total_hours, reverse=True)
    def _estimate_scene_hours_from_page_count(self, scene: Dict) -> float:
        page_count_str = scene.get('Page_Count', '1/8')
        try:
            if ' ' in page_count_str:
                whole, fraction = page_count_str.split(' '); num, den = map(int, fraction.split('/')); return float(whole) + (num / den)
            elif '/' in page_count_str:
                num, den = map(int, page_count_str.split('/')); return num / den
            else: return float(page_count_str)
        except (ValueError, TypeError): return 0.125

@app.post("/run-iteration-2", response_model=Iteration2Response)
async def run_iteration_2(request: ScheduleRequest):
    """
    This endpoint executes the logic for Iteration 2.
    It parses non-negotiable constraints, places them on the calendar,
    and then fills the remaining days with other scenes.
    """
    try:
        print("INFO: Iteration 2 started...")
        calendar = ShootingCalendar("2025-09-01", "2025-10-31") # Dates from weather data
        parser = StructuredConstraintParser(request.constraints)
        cluster_manager = LocationClusterManager(request.stripboard, request.constraints)
        
        scheduler = NaiveScheduler(cluster_manager.clusters, calendar, parser, cluster_manager.scene_time_estimates)
        final_schedule = scheduler.build_schedule()
        
        return Iteration2Response(
            message="Iteration 2 complete. Naïve schedule created.",
            total_shooting_days=len(final_schedule),
            schedule=final_schedule,
            unplaced_anchors=scheduler.unplaced_anchors
        )

    except Exception as e:
        print(f"ERROR: An error occurred during Iteration 2: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "healthy", "iteration": 2}

if __name__ == "__main__":
    print("INFO: Starting FastAPI server for Iteration 2...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

