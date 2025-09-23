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
    constraint_type: str # e.g., 'actor_hard_out', 'location_permit'
    identifier: str # e.g., 'Morgan Freeman', 'University Hospital'
    date: Optional[date] # Can be None for relative constraints like 'shoot_first'
    scenes_affected: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict) # To store extra info


# --- 3. Helper Classes (New for Iteration 2) ---

class ShootingCalendar:
    """Manages the available shooting days for the production."""
    def __init__(self, start_date_str: str, end_date_str: str):
        self.start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        self.end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        self.shooting_days = self._generate_shooting_days()
        print(f"INFO: Calendar created with {len(self.shooting_days)} available shooting days.")

    def _generate_shooting_days(self) -> List[date]:
        """Generates a list of all valid workdays (Mon-Sat) in the date range."""
        days = []
        current_day = self.start_date
        while current_day <= self.end_date:
            # Standard film work week is Monday to Saturday
            if current_day.weekday() != 6: # 6 corresponds to Sunday
                days.append(current_day)
            current_day += timedelta(days=1)
        return days

class StructuredConstraintParser:
    """
    Parses the raw constraints from n8n. In this iteration, it only focuses
    on extracting constraints explicitly marked as "Non-Negotiable".
    """
    def __init__(self, constraints: Dict[str, Any]):
        self.raw_constraints = constraints
        self.non_negotiables = self._parse_non_negotiables()
        print(f"INFO: Parser found {len(self.non_negotiables)} non-negotiable constraints.")

    def _parse_non_negotiables(self) -> List[NonNegotiableConstraint]:
        """Scans all constraint types for those that should be treated as anchors."""
        anchors = []
        
        # --- Actor Hard-Outs ---
        try:
            actors = self.raw_constraints.get('people_constraints', {}).get('actors', {})
            for actor_name, details in actors.items():
                # CORRECTED: Now looks for the exact "Non-negotiable" string.
                if details.get('type') == 'specific_unavailable' and details.get('constraint_level') == 'Non-negotiable':
                    for date_str in details.get('dates', []):
                        anchors.append(NonNegotiableConstraint(
                            constraint_type='actor_hard_out',
                            identifier=actor_name,
                            date=datetime.strptime(date_str, "%Y-%m-%d").date()
                        ))
        except (KeyError, TypeError, ValueError) as e:
            print(f"WARNING: Could not parse actor constraints: {e}")

        # --- Fixed-Date Location Permits (Handles single and multi-day) ---
        try:
            locations = self.raw_constraints.get('location_constraints', {}).get('locations', {})
            for loc_name, details in locations.items():
                for constraint in details.get('constraints', []):
                    if (constraint.get('constraint_type') == 'availability_window' and 
                        constraint.get('constraint_level') == 'Non-negotiable'):
                        
                        parsed_data = constraint['parsed_data']
                        start_date_str = parsed_data.get('start_date')
                        end_date_str = parsed_data.get('end_date')

                        if start_date_str and end_date_str:
                            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
                            
                            # Create an anchor for each day in the non-negotiable range
                            current_date = start_date
                            while current_date <= end_date:
                                anchors.append(NonNegotiableConstraint(
                                    constraint_type='location_permit',
                                    identifier=details.get('real_address', loc_name),
                                    date=current_date,
                                    scenes_affected=details.get('scenes', [])
                                ))
                                current_date += timedelta(days=1)

        except (KeyError, TypeError, ValueError) as e:
            print(f"WARNING: Could not parse location constraints: {e}")

        # --- Director & DOP Notes ---
        try:
            creative = self.raw_constraints.get('creative_constraints', {})
            # Director Notes
            director_notes = creative.get('director_notes', {}).get('director_constraints', [])
            for note in director_notes:
                if note.get('constraint_level') == 'Non-negotiable':
                    anchors.append(NonNegotiableConstraint(
                        constraint_type=f"director_{note.get('constraint_type')}",
                        identifier=note.get('constraint_text', 'Director Note'),
                        date=None, # Relative to schedule, not a fixed date
                        scenes_affected=note.get('related_scenes', []),
                        details=note
                    ))
            # DOP Priorities
            dop_notes = creative.get('dop_priorities', {}).get('dop_priorities', [])
            for note in dop_notes:
                if note.get('constraint_level') == 'Non-negotiable':
                    anchors.append(NonNegotiableConstraint(
                        constraint_type=f"dop_{note.get('category', 'priority')}",
                        identifier=note.get('constraint_text', 'DOP Note'),
                        date=None,
                        scenes_affected=note.get('related_scenes', []),
                        details=note
                    ))
        except (KeyError, TypeError) as e:
            print(f"WARNING: Could not parse creative constraints: {e}")
            
        return anchors


# --- 4. The "Naïve" Scheduler (New for Iteration 2) ---

class NaiveScheduler:
    """
    A simple scheduler that first places anchors and then fills the gaps.
    """
    def __init__(self, clusters: List[LocationCluster], calendar: ShootingCalendar, parser: StructuredConstraintParser, scene_time_estimates: Dict[str, float]):
        self.clusters = clusters
        self.calendar = calendar
        self.parser = parser
        self.schedule: Dict[date, Dict] = {day: {"scenes": [], "location": None} for day in self.calendar.shooting_days}
        self.unplaced_anchors = []
        self.scene_time_estimates = scene_time_estimates

    def build_schedule(self) -> List[Dict[str, Any]]:
        """Main method to construct the naïve schedule."""
        self._place_anchors()
        self._fill_remaining_days()
        return self._format_schedule()

    def _place_anchors(self):
        """Places non-negotiable scenes on their mandatory dates."""
        print("INFO: Placing non-negotiable anchors on the calendar...")
        
        # Sort anchors to handle fixed dates first, then relative ones
        sorted_anchors = sorted(self.parser.non_negotiables, key=lambda x: x.date is None)

        for anchor in sorted_anchors:
            # --- Handle Fixed-Date Anchors ---
            if anchor.date:
                if anchor.date in self.schedule:
                    if self.schedule[anchor.date]['location'] is not None:
                        print(f"WARNING: Date {anchor.date} is already booked. Cannot place anchor for {anchor.identifier}.")
                        self.unplaced_anchors.append(vars(anchor))
                        continue

                    if anchor.constraint_type == 'location_permit':
                        found_cluster = next((c for c in self.clusters if c.location == anchor.identifier), None)
                        if found_cluster:
                            self.schedule[anchor.date]['scenes'].extend(found_cluster.scenes)
                            self.schedule[anchor.date]['location'] = found_cluster.location
                            self.clusters.remove(found_cluster)
                        else:
                            print(f"WARNING: Could not find cluster for anchor location {anchor.identifier}")
                            self.unplaced_anchors.append(vars(anchor))
            
            # --- Handle Relative Anchors (e.g., Shoot First/Last) ---
            else:
                if anchor.constraint_type == 'director_shoot_first':
                    self._place_relative_anchor(anchor, position='first')
                elif anchor.constraint_type == 'director_shoot_last':
                    self._place_relative_anchor(anchor, position='last')
                elif anchor.constraint_type == 'director_same_day_grouping':
                    self._place_relative_anchor(anchor, position='first') # Place same-day groups early by default

    def _place_relative_anchor(self, anchor: NonNegotiableConstraint, position: str):
        """Places scenes for relative anchors like 'shoot_first' or 'shoot_last'."""
        day_to_schedule = None
        if position == 'first':
            # Find the first available day
            for day in self.calendar.shooting_days:
                if self.schedule[day]['location'] is None:
                    day_to_schedule = day
                    break
        elif position == 'last':
            # Find the last available day
            for day in reversed(self.calendar.shooting_days):
                if self.schedule[day]['location'] is None:
                    day_to_schedule = day
                    break

        if day_to_schedule:
            scenes_to_place = []
            location = None
            for scene_num in anchor.scenes_affected:
                for cluster in self.clusters:
                    scene_data = next((s for s in cluster.scenes if str(s.get('Scene_Number')) == str(scene_num)), None)
                    if scene_data:
                        scenes_to_place.append(scene_data)
                        location = cluster.location
                        # Remove the scene so it's not scheduled again
                        cluster.scenes.remove(scene_data)
                        break
            
            if scenes_to_place:
                self.schedule[day_to_schedule]['scenes'].extend(scenes_to_place)
                self.schedule[day_to_schedule]['location'] = location
            else:
                print(f"WARNING: Could not find scenes for relative anchor: {anchor.identifier}")
                self.unplaced_anchors.append(vars(anchor))
        else:
            print(f"WARNING: No available days to place relative anchor: {anchor.identifier}")
            self.unplaced_anchors.append(vars(anchor))

    def _fill_remaining_days(self):
        """Fills empty calendar slots with the remaining location clusters."""
        print("INFO: Filling remaining calendar days sequentially...")
        # Simple strategy: schedule largest remaining clusters first
        sorted_clusters = sorted(self.clusters, key=lambda c: c.total_hours, reverse=True)
        
        day_cursor = 0
        MAX_DAILY_HOURS = 10.0 # Standard assumption for a shooting day

        for cluster in sorted_clusters:
            scenes_to_schedule = list(cluster.scenes)
            while scenes_to_schedule:
                # Find the next available day
                while day_cursor < len(self.calendar.shooting_days) and self.schedule[self.calendar.shooting_days[day_cursor]]['location'] is not None:
                    day_cursor += 1

                if day_cursor >= len(self.calendar.shooting_days):
                    print(f"WARNING: Ran out of calendar days. Cannot schedule all scenes for {cluster.location}.")
                    return # Stop scheduling if calendar is full

                current_date = self.calendar.shooting_days[day_cursor]
                self.schedule[current_date]['location'] = cluster.location
                
                # Pack scenes into this day until it's full
                day_hours = 0
                scenes_for_this_day = []
                while scenes_to_schedule:
                    scene = scenes_to_schedule[0]
                    # CORRECTED: Now uses the time estimates passed directly to the scheduler.
                    scene_hours = self.scene_time_estimates.get(str(scene.get('Scene_Number')), 1.0)
                    
                    if day_hours + scene_hours <= MAX_DAILY_HOURS:
                        scenes_for_this_day.append(scenes_to_schedule.pop(0))
                        day_hours += scene_hours
                    else:
                        break # Day is full
                
                self.schedule[current_date]['scenes'] = scenes_for_this_day

    def _format_schedule(self) -> List[Dict[str, Any]]:
        """Converts the internal schedule dict to a clean list for the API response."""
        formatted_schedule = []
        for day_num, (date_obj, day_data) in enumerate(self.schedule.items()):
            if day_data['scenes']: # Only include days that have scenes scheduled
                formatted_schedule.append({
                    "day": day_num + 1,
                    "date": date_obj.strftime("%Y-%m-%d"),
                    "location": day_data['location'],
                    "scenes": day_data['scenes'],
                    "scene_count": len(day_data['scenes'])
                })
        return formatted_schedule


# --- 5. Main API Endpoint (Updated for Iteration 2) ---

# Re-using the LocationClusterManager from Iteration 1
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
        
        # CORRECTED: Pass the parsed scene_time_estimates from the cluster_manager to the scheduler.
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




