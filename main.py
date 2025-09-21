"""
Film Production Schedule Optimizer - Iteration 1
Goal: Foundational Data Loading & Clustering
This script sets up the API, ingests data from n8n, and groups scenes
into logical, location-based clusters with accurate time estimations.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass
import re
import uvicorn

# --- 1. API and Data Models ---
# Define the structure of the incoming request from n8n and the response.

app = FastAPI(
    title="Film Schedule Optimizer - Iteration 1",
    description="This version focuses on data loading and location clustering only."
)

class ScheduleRequest(BaseModel):
    """
    Defines the expected structure of the JSON data coming from the n8n workflow.
    Pydantic will automatically validate the incoming data against this model.
    """
    stripboard: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    ga_params: Optional[Dict[str, Any]] = {} # Placeholder for future iterations

# A simple response model for this iteration to return the created clusters for testing.
class Iteration1Response(BaseModel):
    message: str
    cluster_count: int
    clusters: List[Dict[str, Any]]


# --- 2. Core Data Structures ---
# Define the primary data structures for organizing the schedule.

@dataclass
class LocationCluster:
    """
    Represents a group of scenes that are all shot at the same physical
    geographic location. This is the fundamental building block for the optimizer.
    """
    location: str
    scenes: List[Dict]
    total_hours: float
    required_actors: List[str]


# --- 3. Location Clustering Logic ---
# This class handles the logic of grouping scenes into clusters.

class LocationClusterManager:
    """
    Groups scenes from the stripboard by their geographic location and calculates
    the total time required to shoot all scenes at that location.
    """
    def __init__(self, stripboard: List[Dict], constraints: Dict[str, Any]):
        self.stripboard = stripboard
        self.scene_time_estimates = self._get_scene_time_estimates(constraints)
        self.clusters = self._create_location_clusters()
        print(f"INFO: Successfully created {len(self.clusters)} location clusters.")

    def _get_scene_time_estimates(self, constraints: Dict) -> Dict[str, float]:
        """
        Extracts the scene time estimates from the nested constraints object.
        This provides the primary source for calculating cluster duration.
        """
        try:
            time_estimates = constraints['operational_data']['time_estimates']['scene_estimates']
            
            # The scene number key can have a weird prefix, so we check for any key containing "Scene_Number"
            # and map it to the 'Estimated_Time_Hours'.
            return {
                str(est[key]): float(est['Estimated_Time_Hours'])
                for est in time_estimates
                for key in est if 'Scene_Number' in key
            }
        except (KeyError, TypeError):
            print("WARNING: Could not find 'scene_time_estimates' in constraints. Will use fallback page count estimates.")
            return {}

    def _create_location_clusters(self) -> List[LocationCluster]:
        """
        Iterates through the stripboard, groups scenes by 'Geographic_Location',
        and calculates total hours and required actors for each cluster.
        """
        location_groups = defaultdict(list)
        for scene in self.stripboard:
            location = scene.get('Geographic_Location')
            if location and location != 'Location TBD':
                location_groups[location].append(scene)

        clusters = []
        for location, scenes in location_groups.items():
            total_hours = 0.0
            all_actors = set()

            for scene in scenes:
                scene_number = str(scene.get('Scene_Number', ''))
                
                # Use real time estimate if available, otherwise use fallback
                if scene_number in self.scene_time_estimates:
                    scene_hours = self.scene_time_estimates[scene_number]
                else:
                    scene_hours = self._estimate_scene_hours_from_page_count(scene)
                
                total_hours += scene_hours
                
                # Collect all unique actors required for this location
                cast = scene.get('Cast', [])
                if isinstance(cast, list):
                    all_actors.update(cast)

            clusters.append(LocationCluster(
                location=location,
                scenes=sorted(scenes, key=lambda s: s.get('Scene_Number', '')), # Keep scenes numerically sorted
                total_hours=round(total_hours, 2),
                required_actors=sorted(list(all_actors))
            ))
            
        # Sort clusters by total hours (largest first) for potential future scheduling logic
        return sorted(clusters, key=lambda c: c.total_hours, reverse=True)

    def _estimate_scene_hours_from_page_count(self, scene: Dict) -> float:
        """
        Fallback method to estimate scene duration based on script page count.
        This is used when a real time estimate is not available.
        """
        page_count_str = scene.get('Page_Count', '1/8')
        try:
            if ' ' in page_count_str: # Handles "1 1/8"
                whole, fraction = page_count_str.split(' ')
                num, den = map(int, fraction.split('/'))
                return float(whole) + (num / den)
            elif '/' in page_count_str: # Handles "1/8"
                num, den = map(int, page_count_str.split('/'))
                return num / den
            else: # Handles "1"
                return float(page_count_str)
        except (ValueError, TypeError):
            return 0.125 # Default to 1/8 page if parsing fails

# --- 4. Main API Endpoint ---

@app.post("/run-iteration-1", response_model=Iteration1Response)
async def run_iteration_1(request: ScheduleRequest):
    """
    This endpoint executes the logic for Iteration 1.
    It receives the full JSON from n8n, creates the location clusters,
    and returns them for testing and validation.
    """
    try:
        print("INFO: Iteration 1 started. Initializing LocationClusterManager...")
        cluster_manager = LocationClusterManager(request.stripboard, request.constraints)
        
        # Convert dataclass objects to dictionaries for the JSON response
        clusters_as_dicts = [
            {
                "location": cluster.location,
                "total_hours": cluster.total_hours,
                "scene_count": len(cluster.scenes),
                "required_actors": cluster.required_actors,
                "scenes": cluster.scenes
            }
            for cluster in cluster_manager.clusters
        ]

        return Iteration1Response(
            message="Iteration 1 complete. Location clusters created successfully.",
            cluster_count=len(clusters_as_dicts),
            clusters=clusters_as_dicts
        )

    except Exception as e:
        print(f"ERROR: An error occurred during Iteration 1: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. Health Check and Runner ---

@app.get("/")
async def health_check():
    return {"status": "healthy", "iteration": 1}

if __name__ == "__main__":
    print("INFO: Starting FastAPI server for Iteration 1...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
