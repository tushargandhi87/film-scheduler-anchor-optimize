"""
Film Production Scheduling Optimizer - Iteration 1
Foundational Data Loading & Location Clustering

This iteration implements:
1. JSON data validation and ingestion
2. Date-specific non-negotiable constraint anchoring
3. LocationClusterManager for geographic scene clustering
4. Shooting time calculation per location cluster
5. FastAPI web service for deployment on Railway
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SceneInfo:
    """Data structure for individual scene information"""
    scene_number: str
    int_ext: str
    location_name: str
    day_night: str
    synopsis: str
    page_count: str
    script_day: str
    cast: List[str]
    geographic_location: str
    estimated_time_hours: float = 0.0
    complexity_tier: str = ""

@dataclass
class DateAnchor:
    """Data structure for date-specific non-negotiable constraints"""
    constraint_type: str  # 'actor_departure', 'location_permit', 'equipment_return'
    entity_name: str      # Actor name, location name, etc.
    anchor_date: datetime
    constraint_details: str
    affected_scenes: List[str] = None

@dataclass
class LocationCluster:
    """Data structure for geographic location clusters"""
    geographic_location: str
    scenes: List[SceneInfo]
    total_shooting_hours: float
    total_shooting_days: int
    complexity_distribution: Dict[str, int]
    cast_requirements: set

class DataValidator:
    """Validates incoming JSON data structure"""
    
    @staticmethod
    def validate_input(data: Any) -> bool:
        """Validate the input data structure matches expected schema"""
        try:
            if not isinstance(data, list) or len(data) != 1:
                raise ValueError("Input must be an array with exactly one object")
            
            production_data = data[0]
            required_keys = ['stripboard', 'constraints', 'ga_params']
            
            for key in required_keys:
                if key not in production_data:
                    raise ValueError(f"Missing required key: {key}")
            
            # Validate stripboard structure
            stripboard = production_data['stripboard']
            if not isinstance(stripboard, list):
                raise ValueError("Stripboard must be an array")
            
            # Validate at least one scene exists
            if len(stripboard) == 0:
                raise ValueError("Stripboard cannot be empty")
            
            # Validate required scene fields
            required_scene_fields = [
                'Scene_Number', 'INT_EXT', 'Location_Name', 'Day_Night',
                'Synopsis', 'Page_Count', 'Script_Day', 'Cast', 'Geographic_Location'
            ]
            
            for scene in stripboard:
                for field in required_scene_fields:
                    if field not in scene:
                        raise ValueError(f"Scene missing required field: {field}")
            
            logger.info(f"Data validation successful: {len(stripboard)} scenes loaded")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False

class DateAnchorExtractor:
    """Extracts date-specific non-negotiable constraints"""
    
    def __init__(self, constraints_data: Dict):
        self.constraints = constraints_data
        self.date_anchors: List[DateAnchor] = []
    
    def extract_all_anchors(self) -> List[DateAnchor]:
        """Extract all date-specific non-negotiable constraints"""
        self.date_anchors = []
        
        # Extract actor date constraints
        self._extract_actor_date_constraints()
        
        # Extract location permit constraints
        self._extract_location_date_constraints()
        
        # Extract production rule date constraints
        self._extract_production_rule_constraints()
        
        logger.info(f"Extracted {len(self.date_anchors)} date-specific anchors")
        return self.date_anchors
    
    def _extract_actor_date_constraints(self):
        """Extract actor availability constraints with specific dates"""
        try:
            people_constraints = self.constraints.get('people_constraints', {})
            actors = people_constraints.get('actors', {})
            
            for actor_id, actor_data in actors.items():
                constraint_level = actor_data.get('constraint_level', '')
                if constraint_level.lower() == 'non-negotiable':
                    dates = actor_data.get('dates', [])
                    constraint_type = actor_data.get('type', '')
                    
                    for date_str in dates:
                        try:
                            anchor_date = datetime.strptime(date_str, '%Y-%m-%d')
                            anchor = DateAnchor(
                                constraint_type=f'actor_{constraint_type}',
                                entity_name=actor_id,
                                anchor_date=anchor_date,
                                constraint_details=actor_data.get('notes', ''),
                                affected_scenes=[]
                            )
                            self.date_anchors.append(anchor)
                            logger.info(f"Added actor anchor: {actor_id} - {constraint_type} on {date_str}")
                        except ValueError as e:
                            logger.warning(f"Invalid date format for actor {actor_id}: {date_str}")
                            
        except Exception as e:
            logger.error(f"Error extracting actor constraints: {str(e)}")
    
    def _extract_location_date_constraints(self):
        """Extract location constraints with specific dates"""
        try:
            location_constraints = self.constraints.get('location_constraints', {})
            locations = location_constraints.get('locations', {})
            
            for location_id, location_data in locations.items():
                constraints = location_data.get('constraints', [])
                
                for constraint in constraints:
                    if constraint.get('constraint_level', '').lower() == 'non-negotiable':
                        # Look for date-specific constraints in the constraint data
                        constraint_text = str(constraint.get('raw_text', ''))
                        if 'date' in constraint_text.lower() or 'deadline' in constraint_text.lower():
                            # This is a simplified extraction - in real implementation,
                            # would need more sophisticated date parsing
                            logger.info(f"Found potential date constraint for location {location_id}")
                            
        except Exception as e:
            logger.error(f"Error extracting location constraints: {str(e)}")
    
    def _extract_production_rule_constraints(self):
        """Extract production rules with date-specific requirements"""
        try:
            operational_data = self.constraints.get('operational_data', {})
            production_rules = operational_data.get('production_rules', {})
            rules = production_rules.get('rules', [])
            
            for rule in rules:
                constraint_level = rule.get('constraint_level', '')
                if constraint_level.lower() == 'non-negotiable':
                    rule_text = rule.get('raw_text', '')
                    if 'date' in rule_text.lower() or 'deadline' in rule_text.lower():
                        logger.info(f"Found potential date-specific production rule: {rule.get('parameter_name', 'Unknown')}")
                        
        except Exception as e:
            logger.error(f"Error extracting production rule constraints: {str(e)}")

class LocationClusterManager:
    """Manages geographic clustering of scenes and calculates shooting requirements"""
    
    def __init__(self, stripboard_data: List[Dict], time_estimates_data: List[Dict]):
        self.stripboard = stripboard_data
        self.time_estimates = self._create_time_estimates_lookup(time_estimates_data)
        self.location_clusters: Dict[str, LocationCluster] = {}
        self.scenes: List[SceneInfo] = []
        
    def _create_time_estimates_lookup(self, time_estimates_data: List[Dict]) -> Dict[str, Dict]:
        """Create a lookup dictionary for time estimates by scene number"""
        lookup = {}
        for estimate in time_estimates_data:
            scene_number = estimate.get('Scene_Number', '')
            lookup[scene_number] = estimate
        return lookup
    
    def _parse_page_count(self, page_count_str: str) -> float:
        """Parse page count string to float (handles fractions like '2 3/8')"""
        try:
            # Handle simple numbers
            if '/' not in page_count_str:
                return float(page_count_str)
            
            # Handle fractions like '2 3/8' or '3/8'
            parts = page_count_str.strip().split()
            if len(parts) == 2:  # '2 3/8'
                whole = float(parts[0])
                fraction_parts = parts[1].split('/')
                fraction = float(fraction_parts[0]) / float(fraction_parts[1])
                return whole + fraction
            elif len(parts) == 1 and '/' in parts[0]:  # '3/8'
                fraction_parts = parts[0].split('/')
                return float(fraction_parts[0]) / float(fraction_parts[1])
            else:
                return float(page_count_str)
        except:
            logger.warning(f"Could not parse page count: {page_count_str}, defaulting to 1.0")
            return 1.0
    
    def _create_scene_info(self, scene_data: Dict) -> SceneInfo:
        """Create SceneInfo object from scene data"""
        scene_number = scene_data['Scene_Number']
        
        # Get time estimate from lookup
        estimated_hours = 0.0
        complexity_tier = ""
        
        if scene_number in self.time_estimates:
            estimate_data = self.time_estimates[scene_number]
            try:
                estimated_hours = float(estimate_data.get('Estimated_Time_Hours', '0'))
            except:
                # Fallback to page-based estimation (roughly 1 page = 1 hour)
                page_count = self._parse_page_count(scene_data.get('Page_Count', '1'))
                estimated_hours = page_count
            
            complexity_tier = estimate_data.get('Complexity_Tier', 'Medium')
        else:
            # Fallback estimation based on page count
            page_count = self._parse_page_count(scene_data.get('Page_Count', '1'))
            estimated_hours = page_count
            complexity_tier = 'Medium'
        
        return SceneInfo(
            scene_number=scene_number,
            int_ext=scene_data['INT_EXT'],
            location_name=scene_data['Location_Name'],
            day_night=scene_data['Day_Night'],
            synopsis=scene_data['Synopsis'],
            page_count=scene_data['Page_Count'],
            script_day=scene_data['Script_Day'],
            cast=scene_data['Cast'],
            geographic_location=scene_data['Geographic_Location'],
            estimated_time_hours=estimated_hours,
            complexity_tier=complexity_tier
        )
    
    def cluster_scenes_by_location(self) -> Dict[str, LocationCluster]:
        """Group scenes by geographic location and calculate cluster metrics"""
        # Convert stripboard to SceneInfo objects
        self.scenes = [self._create_scene_info(scene) for scene in self.stripboard]
        
        # Group scenes by geographic location
        location_groups = defaultdict(list)
        for scene in self.scenes:
            location_groups[scene.geographic_location].append(scene)
        
        # Create LocationCluster objects
        for location, scenes in location_groups.items():
            total_hours = sum(scene.estimated_time_hours for scene in scenes)
            
            # Calculate shooting days (assuming 10-hour shooting days)
            total_days = max(1, int(total_hours / 10) + (1 if total_hours % 10 > 0 else 0))
            
            # Analyze complexity distribution
            complexity_counts = defaultdict(int)
            for scene in scenes:
                complexity_counts[scene.complexity_tier] += 1
            
            # Collect all cast requirements
            cast_requirements = set()
            for scene in scenes:
                cast_requirements.update(scene.cast)
            
            cluster = LocationCluster(
                geographic_location=location,
                scenes=scenes,
                total_shooting_hours=total_hours,
                total_shooting_days=total_days,
                complexity_distribution=dict(complexity_counts),
                cast_requirements=cast_requirements
            )
            
            self.location_clusters[location] = cluster
            
            logger.info(f"Location cluster '{location}': {len(scenes)} scenes, "
                       f"{total_hours:.1f} hours, {total_days} days")
        
        return self.location_clusters
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for all location clusters"""
        total_scenes = sum(len(cluster.scenes) for cluster in self.location_clusters.values())
        total_hours = sum(cluster.total_shooting_hours for cluster in self.location_clusters.values())
        total_days = sum(cluster.total_shooting_days for cluster in self.location_clusters.values())
        
        return {
            'total_locations': len(self.location_clusters),
            'total_scenes': total_scenes,
            'total_shooting_hours': total_hours,
            'total_shooting_days': total_days,
            'locations_by_size': sorted(
                [(loc, cluster.total_shooting_days) for loc, cluster in self.location_clusters.items()],
                key=lambda x: x[1],
                reverse=True
            )
        }

class ProductionScheduler:
    """Main scheduler class that coordinates all components"""
    
    def __init__(self, input_data: List[Dict]):
        if not DataValidator.validate_input(input_data):
            raise ValueError("Input data validation failed")
        
        self.production_data = input_data[0]
        self.stripboard = self.production_data['stripboard']
        self.constraints = self.production_data['constraints']
        self.ga_params = self.production_data['ga_params']
        
        # Initialize components
        self.date_anchor_extractor = DateAnchorExtractor(self.constraints)
        
        # Get time estimates data
        time_estimates_data = (
            self.constraints
            .get('operational_data', {})
            .get('time_estimates', {})
            .get('scene_estimates', [])
        )
        
        self.location_manager = LocationClusterManager(self.stripboard, time_estimates_data)
        
        # Initialize results storage
        self.date_anchors: List[DateAnchor] = []
        self.location_clusters: Dict[str, LocationCluster] = {}
    
    def process_iteration_1(self) -> Dict[str, Any]:
        """Execute Iteration 1: Data loading, anchoring, and clustering"""
        logger.info("Starting Iteration 1: Foundational Data Loading & Location Clustering")
        
        # Step 1: Extract date-specific non-negotiable anchors
        logger.info("Step 1: Extracting date-specific anchors...")
        self.date_anchors = self.date_anchor_extractor.extract_all_anchors()
        
        # Step 2: Create location clusters
        logger.info("Step 2: Creating location clusters...")
        self.location_clusters = self.location_manager.cluster_scenes_by_location()
        
        # Step 3: Generate summary
        logger.info("Step 3: Generating summary...")
        cluster_summary = self.location_manager.get_cluster_summary()
        
        # Compile results
        results = {
            'iteration': 1,
            'status': 'completed',
            'date_anchors': [
                {
                    'constraint_type': anchor.constraint_type,
                    'entity_name': anchor.entity_name,
                    'anchor_date': anchor.anchor_date.isoformat(),
                    'constraint_details': anchor.constraint_details
                }
                for anchor in self.date_anchors
            ],
            'location_clusters': {
                location: {
                    'scene_count': len(cluster.scenes),
                    'total_hours': cluster.total_shooting_hours,
                    'total_days': cluster.total_shooting_days,
                    'complexity_distribution': cluster.complexity_distribution,
                    'cast_count': len(cluster.cast_requirements),
                    'scenes': [scene.scene_number for scene in cluster.scenes]
                }
                for location, cluster in self.location_clusters.items()
            },
            'summary': cluster_summary
        }
        
        logger.info(f"Iteration 1 completed successfully: "
                   f"{len(self.date_anchors)} anchors, "
                   f"{len(self.location_clusters)} location clusters")
        
        return results

# FastAPI Application
app = FastAPI(
    title="Film Production Scheduling Optimizer",
    description="AI-powered film production scheduling with hierarchical constraint optimization",
    version="1.0.0"
)

class ScheduleResponse(BaseModel):
    """Pydantic model for the schedule response"""
    success: bool
    iteration: int
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_seconds: Optional[float] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Film Production Scheduling Optimizer",
        "status": "active",
        "iteration": 1,
        "description": "Foundational Data Loading & Location Clustering"
    }

@app.get("/health")
async def health_check():
    """Detailed health check for Railway deployment"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "film-scheduler",
        "iteration": 1
    }

@app.post("/schedule", response_model=ScheduleResponse)
async def create_schedule(request: List[Dict[str, Any]]):
    """
    Main endpoint to process film production scheduling
    Expects JSON payload matching the http_request.schema.json structure (array directly)
    """
    start_time = datetime.now()
    
    try:
        logger.info("Received scheduling request")
        logger.info(f"Input data size: {len(request)} objects")
        
        # Initialize scheduler with the input data
        scheduler = ProductionScheduler(request)
        
        # Process Iteration 1
        results = scheduler.process_iteration_1()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return ScheduleResponse(
            success=True,
            iteration=1,
            data=results,
            processing_time_seconds=processing_time
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return ScheduleResponse(
            success=False,
            iteration=1,
            error=f"Data validation failed: {str(e)}",
            processing_time_seconds=(datetime.now() - start_time).total_seconds()
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ScheduleResponse(
            success=False,
            iteration=1,
            error=f"Processing failed: {str(e)}",
            processing_time_seconds=(datetime.now() - start_time).total_seconds()
        )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Invalid request format",
            "details": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "details": str(exc)
        }
    )

def main():
    """Main entry point for local testing"""
    try:
        # Example usage for local testing
        sample_data = [{
            'stripboard': [
                {
                    'Scene_Number': '1',
                    'INT_EXT': 'INT',
                    'Location_Name': 'Kitchen',
                    'Day_Night': 'DAY',
                    'Synopsis': 'Character A makes breakfast',
                    'Page_Count': '2',
                    'Script_Day': '1',
                    'Cast': ['Actor1', 'Actor2'],
                    'Geographic_Location': 'House_Interior'
                }
            ],
            'constraints': {
                'people_constraints': {'actors': {}},
                'operational_data': {'time_estimates': {'scene_estimates': []}},
                'creative_constraints': {},
                'location_constraints': {},
                'technical_constraints': {}
            },
            'ga_params': {
                'phase1_population': 100,
                'phase1_generations': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'seed': 42,
                'conflict_tolerance': 0.05
            }
        }]
        
        scheduler = ProductionScheduler(sample_data)
        results = scheduler.process_iteration_1()
        
        print("=== Iteration 1 Results ===")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # For Railway deployment, use uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
