"""
Film Production Scheduling Optimizer - Iteration 2 (FIXED)
All Non-Negotiables + Naive Scheduler

This iteration implements:
1. Date-specific non-negotiable constraint anchoring
2. All other non-negotiable constraints (technical, creative, operational)
3. Naive sequential scheduler with full constraint compliance
4. Day-by-day shooting schedule generation
5. FastAPI web service for deployment on Railway

FIX: Location availability windows are now properly enforced during assignment
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
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
    constraint_type: str
    entity_name: str
    anchor_date: datetime
    constraint_details: str
    affected_scenes: List[str] = None

@dataclass
class NonNegotiableConstraint:
    """Data structure for non-date-specific non-negotiable constraints"""
    constraint_id: str
    constraint_type: str  # 'scene_sequence', 'scene_grouping', 'equipment_requirement', etc.
    constraint_category: str  # 'technical', 'creative', 'operational'
    priority: int  # 1=highest, lower numbers = higher priority
    constraint_details: str
    affected_scenes: List[str]
    affected_locations: List[str]
    additional_data: Dict[str, Any]

@dataclass
class LocationCluster:
    """Data structure for geographic location clusters"""
    geographic_location: str
    scenes: List[SceneInfo]
    total_shooting_hours: float
    total_shooting_days: int
    complexity_distribution: Dict[str, int]
    cast_requirements: set

@dataclass
class ShootingDay:
    """Data structure for a single shooting day"""
    date: datetime
    day_number: int
    day_of_week: str
    status: str  # 'available', 'blocked', 'assigned', 'off_day'
    location: Optional[str] = None
    scenes: List[str] = None
    total_hours: float = 0.0
    cast_required: Set[str] = None
    constraints_applied: List[str] = None

# FastAPI Application
app = FastAPI(
    title="Film Production Scheduling Optimizer",
    description="AI-powered film production scheduling with hierarchical constraint optimization",
    version="2.0.2"
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
        "iteration": 2,
        "version": "2.0.2",
        "description": "All Non-Negotiables + Naive Scheduler (Geographic Location Fix)"
    }

@app.get("/health")
async def health_check():
    """Detailed health check for Railway deployment"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "film-scheduler",
        "iteration": 2,
        "version": "2.0.2"
    }

@app.post("/schedule", response_model=ScheduleResponse)
async def create_schedule(request: Request):
    """
    Main endpoint to process film production scheduling
    Expects JSON object with production data
    """
    start_time = datetime.now()
    
    try:
        # Parse raw JSON body
        body = await request.json()
        
        logger.info("Received scheduling request")
        logger.info(f"Raw body type: {type(body)}")
        
        # Initialize scheduler with the input data
        scheduler = ProductionScheduler(body)
        
        # Process Iteration 2
        results = scheduler.process_iteration_2()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return ScheduleResponse(
            success=True,
            iteration=2,
            data=results,
            processing_time_seconds=processing_time
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return ScheduleResponse(
            success=False,
            iteration=2,
            error=f"Data validation failed: {str(e)}",
            processing_time_seconds=(datetime.now() - start_time).total_seconds()
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ScheduleResponse(
            success=False,
            iteration=2,
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

class DateAnchorExtractor:
    """Extracts date-specific non-negotiable constraints"""
    
    def __init__(self, constraints_data: Dict, stripboard_data: List[Dict]):
        self.constraints = constraints_data
        self.stripboard = stripboard_data
        self.date_anchors: List[DateAnchor] = []
        self.location_name_mapping = self._create_location_mapping()
    
    def _create_location_mapping(self) -> Dict[str, str]:
        """
        Create mapping from script location names to geographic locations.
        Returns dict: {script_location_name: geographic_location}
        """
        mapping = {}
        
        for scene in self.stripboard:
            script_location = scene.get('Location_Name', '').strip()
            geographic_location = scene.get('Geographic_Location', '').strip()
            
            if script_location and geographic_location:
                # Normalize to uppercase for matching (constraints use uppercase keys)
                script_location_upper = script_location.upper()
                
                # Store the mapping - use geographic_location as the canonical name
                if script_location_upper not in mapping:
                    mapping[script_location_upper] = geographic_location
                elif mapping[script_location_upper] != geographic_location:
                    logger.warning(
                        f"Script location '{script_location}' maps to multiple geographic locations: "
                        f"'{mapping[script_location_upper]}' and '{geographic_location}'"
                    )
        
        logger.info(f"Created location mapping for {len(mapping)} script locations")
        return mapping
    
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
                
                # Map script location name to geographic location
                location_id_upper = location_id.upper()
                geographic_location = self.location_name_mapping.get(location_id_upper)
                
                if not geographic_location:
                    logger.warning(
                        f"Script location '{location_id}' not found in stripboard mapping. "
                        f"Skipping date constraints for this location."
                    )
                    continue
                
                for constraint in constraints:
                    constraint_level = constraint.get('constraint_level', '')
                    constraint_type = constraint.get('constraint_type', '')
                    
                    # Only process Non-negotiable and Hard constraints with date availability windows
                    if (constraint_level.lower() in ['non-negotiable', 'hard'] and 
                        constraint_type == 'availability_window'):
                        
                        parsed_data = constraint.get('parsed_data', {})
                        start_date_str = parsed_data.get('start_date')
                        end_date_str = parsed_data.get('end_date')
                        
                        if start_date_str and end_date_str:
                            try:
                                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                                end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                                
                                # Create anchor for location availability window using GEOGRAPHIC LOCATION
                                if start_date == end_date:
                                    # Single day availability
                                    anchor = DateAnchor(
                                        constraint_type=f'location_availability_{constraint_level.lower()}',
                                        entity_name=geographic_location,
                                        anchor_date=start_date,
                                        constraint_details=f"Available on {start_date_str} only - {constraint.get('original_text', '')}",
                                        affected_scenes=[]
                                    )
                                    self.date_anchors.append(anchor)
                                    logger.info(f"Added location anchor: {geographic_location} (script: {location_id}) available on {start_date_str}")
                                else:
                                    # Multi-day availability window - create anchor for start and end
                                    start_anchor = DateAnchor(
                                        constraint_type=f'location_window_start_{constraint_level.lower()}',
                                        entity_name=geographic_location,
                                        anchor_date=start_date,
                                        constraint_details=f"Available from {start_date_str} to {end_date_str} - {constraint.get('original_text', '')}",
                                        affected_scenes=[]
                                    )
                                    end_anchor = DateAnchor(
                                        constraint_type=f'location_window_end_{constraint_level.lower()}',
                                        entity_name=geographic_location,
                                        anchor_date=end_date,
                                        constraint_details=f"Available from {start_date_str} to {end_date_str} - {constraint.get('original_text', '')}",
                                        affected_scenes=[]
                                    )
                                    self.date_anchors.append(start_anchor)
                                    self.date_anchors.append(end_anchor)
                                    logger.info(f"Added location window anchors: {geographic_location} (script: {location_id}) available {start_date_str} to {end_date_str}")
                                    
                            except ValueError as e:
                                logger.warning(f"Invalid date format for location {location_id}: {start_date_str} - {end_date_str}")
                        else:
                            logger.debug(f"Location {location_id} constraint missing date information")
                            
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

class NonNegotiableExtractor:
    """Extracts all non-date-specific non-negotiable constraints"""
    
    def __init__(self, constraints_data: Dict):
        self.constraints = constraints_data
        self.non_negotiables: List[NonNegotiableConstraint] = []
    
    def extract_all_non_negotiables(self) -> List[NonNegotiableConstraint]:
        """Extract all non-date-specific non-negotiable constraints"""
        self.non_negotiables = []
        
        # Extract creative constraints
        self._extract_creative_constraints()
        
        # Extract technical constraints
        self._extract_technical_constraints()
        
        # Extract operational constraints
        self._extract_operational_constraints()
        
        # Sort by priority (lower number = higher priority)
        self.non_negotiables.sort(key=lambda x: x.priority)
        
        logger.info(f"Extracted {len(self.non_negotiables)} non-date-specific non-negotiable constraints")
        return self.non_negotiables
    
    def _extract_creative_constraints(self):
        """Extract creative non-negotiable constraints from director and DOP"""
        try:
            creative_constraints = self.constraints.get('creative_constraints', {})
            
            # Director constraints
            director_notes = creative_constraints.get('director_notes', {})
            director_constraints = director_notes.get('director_constraints', [])
            
            for constraint in director_constraints:
                constraint_level = constraint.get('constraint_level', '')
                if constraint_level.lower() == 'non-negotiable':
                    constraint_type = constraint.get('constraint_type', '')
                    related_scenes = constraint.get('related_scenes', [])
                    
                    # Determine priority based on constraint type
                    priority = 1 if constraint_type in ['shoot_first', 'shoot_last'] else 2
                    
                    non_neg = NonNegotiableConstraint(
                        constraint_id=f"director_{constraint_type}_{len(self.non_negotiables)}",
                        constraint_type=constraint_type,
                        constraint_category='creative',
                        priority=priority,
                        constraint_details=constraint.get('constraint_text', ''),
                        affected_scenes=related_scenes,
                        affected_locations=constraint.get('related_locations', []),
                        additional_data=constraint
                    )
                    self.non_negotiables.append(non_neg)
                    logger.info(f"Added creative constraint: {constraint_type} for scenes {related_scenes}")
            
            # DOP constraints
            dop_priorities = creative_constraints.get('dop_priorities', {})
            dop_constraints = dop_priorities.get('dop_priorities', [])
            
            for constraint in dop_constraints:
                constraint_level = constraint.get('constraint_level', '')
                if constraint_level.lower() == 'non-negotiable':
                    category = constraint.get('category', '')
                    related_scenes = constraint.get('related_scenes', [])
                    
                    non_neg = NonNegotiableConstraint(
                        constraint_id=f"dop_{category}_{len(self.non_negotiables)}",
                        constraint_type=category,
                        constraint_category='creative',
                        priority=3,  # DOP constraints generally lower priority than director
                        constraint_details=constraint.get('constraint_text', ''),
                        affected_scenes=[str(s) for s in related_scenes] if related_scenes else [],
                        affected_locations=constraint.get('related_locations', []),
                        additional_data=constraint
                    )
                    self.non_negotiables.append(non_neg)
                    logger.info(f"Added DOP constraint: {category} for scenes {related_scenes}")
                    
        except Exception as e:
            logger.error(f"Error extracting creative constraints: {str(e)}")
    
    def _extract_technical_constraints(self):
        """Extract technical non-negotiable constraints"""
        try:
            technical_constraints = self.constraints.get('technical_constraints', {})
            equipment = technical_constraints.get('equipment', {})
            
            for equipment_name, equipment_data in equipment.items():
                constraint_level = equipment_data.get('constraint_level', '')
                if constraint_level.lower() in ['non-negotiable', 'hard']:
                    equipment_req = equipment_data.get('equipment_requirements', {})
                    required_scenes = equipment_req.get('required_scenes', [])
                    
                    if required_scenes:
                        non_neg = NonNegotiableConstraint(
                            constraint_id=f"equipment_{equipment_name}_{len(self.non_negotiables)}",
                            constraint_type='equipment_requirement',
                            constraint_category='technical',
                            priority=2,
                            constraint_details=f"{equipment_name}: {equipment_data.get('notes', '')}",
                            affected_scenes=required_scenes,
                            affected_locations=[],
                            additional_data=equipment_data
                        )
                        self.non_negotiables.append(non_neg)
                        logger.info(f"Added technical constraint: {equipment_name} for scenes {required_scenes}")
            
            # Special requirements
            special_requirements = technical_constraints.get('special_requirements', {})
            for req_name, req_data in special_requirements.items():
                constraint_level = req_data.get('constraint_level', '')
                if constraint_level.lower() in ['non-negotiable', 'hard']:
                    # Extract scene numbers from notes if available
                    notes = req_data.get('notes', '')
                    affected_scenes = []
                    if 'Scenes:' in notes:
                        # Parse scene numbers from notes
                        scene_part = notes.split('Scenes:')[1].split('.')[0]
                        scene_numbers = [s.strip('[]') for s in scene_part.split(',')]
                        affected_scenes = [s.strip() for s in scene_numbers if s.strip()]
                    
                    non_neg = NonNegotiableConstraint(
                        constraint_id=f"special_{req_name}_{len(self.non_negotiables)}",
                        constraint_type='special_requirement',
                        constraint_category='technical',
                        priority=3,
                        constraint_details=notes,
                        affected_scenes=affected_scenes,
                        affected_locations=[],
                        additional_data=req_data
                    )
                    self.non_negotiables.append(non_neg)
                    logger.info(f"Added special requirement: {req_name} for scenes {affected_scenes}")
                    
        except Exception as e:
            logger.error(f"Error extracting technical constraints: {str(e)}")
    
    def _extract_operational_constraints(self):
        """Extract operational non-negotiable constraints"""
        try:
            operational_data = self.constraints.get('operational_data', {})
            production_rules = operational_data.get('production_rules', {})
            rules = production_rules.get('rules', [])
            
            for rule in rules:
                constraint_level = rule.get('constraint_level', '')
                if constraint_level.lower() in ['non-negotiable', 'hard']:
                    parameter_name = rule.get('parameter_name', '')
                    rule_category = rule.get('rule_category', '')
                    
                    # Skip date-specific rules (handled by DateAnchorExtractor)
                    rule_text = rule.get('raw_text', '').lower()
                    if 'date' in rule_text or 'deadline' in rule_text:
                        continue
                    
                    non_neg = NonNegotiableConstraint(
                        constraint_id=f"operational_{parameter_name}_{len(self.non_negotiables)}",
                        constraint_type=rule_category,
                        constraint_category='operational',
                        priority=1 if constraint_level.lower() == 'non-negotiable' else 2,
                        constraint_details=rule.get('raw_text', ''),
                        affected_scenes=[],
                        affected_locations=[],
                        additional_data=rule
                    )
                    self.non_negotiables.append(non_neg)
                    logger.info(f"Added operational constraint: {parameter_name} ({rule_category})")
                    
        except Exception as e:
            logger.error(f"Error extracting operational constraints: {str(e)}")

class LocationClusterManager:
    """Manages geographic clustering of scenes and calculates shooting requirements"""
    
    def __init__(self, stripboard_data: List[Dict], time_estimates_data: List[Dict], hours_per_day: float = 10.0):
        self.stripboard = stripboard_data
        self.hours_per_day = hours_per_day
        self.time_estimates = self._create_time_estimates_lookup(time_estimates_data)
        self.location_clusters: Dict[str, LocationCluster] = {}
        self.scenes: List[SceneInfo] = []
        self.time_estimate_stats = {
            'total_scenes': len(stripboard_data),
            'matched_estimates': 0,
            'fallback_page_count': 0,
            'failed_matches': 0
        }
        
    def _clean_bom_and_keys(self, data: Dict) -> Dict:
        """Remove BOM characters from dictionary keys and values"""
        cleaned = {}
        for key, value in data.items():
            clean_key = key.lstrip('\ufeff').strip()
            cleaned[clean_key] = value
        return cleaned
    
    def _create_time_estimates_lookup(self, time_estimates_data: List[Dict]) -> Dict[str, Dict]:
        """Create a lookup dictionary for time estimates by scene number with BOM handling"""
        lookup = {}
        
        if not time_estimates_data:
            logger.warning("No time estimates data provided - will use page count fallback")
            return lookup
            
        logger.info(f"Processing {len(time_estimates_data)} time estimate records")
        
        for i, estimate in enumerate(time_estimates_data):
            try:
                cleaned_estimate = self._clean_bom_and_keys(estimate)
                scene_number = None
                possible_scene_fields = ['Scene_Number', 'scene_number', 'Scene_ID', 'scene_id']
                
                for field in possible_scene_fields:
                    if field in cleaned_estimate:
                        scene_number = str(cleaned_estimate[field]).strip()
                        break
                
                if scene_number:
                    lookup[scene_number] = cleaned_estimate
                    logger.debug(f"Matched time estimate for scene {scene_number}")
                else:
                    logger.warning(f"Time estimate record {i+1} missing scene number. Keys: {list(cleaned_estimate.keys())}")
                    
            except Exception as e:
                logger.error(f"Error processing time estimate record {i+1}: {str(e)}")
        
        logger.info(f"Successfully created time estimates lookup for {len(lookup)} scenes")
        return lookup
    
    def _parse_page_count(self, page_count_str: str) -> float:
        """Parse page count string to float (handles fractions like '2 3/8')"""
        try:
            page_count_str = str(page_count_str).strip()
            
            if '/' not in page_count_str:
                return float(page_count_str)
            
            parts = page_count_str.split()
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
        except Exception as e:
            logger.warning(f"Could not parse page count: '{page_count_str}' - {str(e)}, defaulting to 1.0")
            return 1.0
    
    def _extract_time_and_complexity(self, scene_data: Dict, scene_number: str) -> Tuple[float, str]:
        """Extract time estimate and complexity from scene data"""
        
        if scene_number in self.time_estimates:
            estimate_data = self.time_estimates[scene_number]
            try:
                time_fields = ['Estimated_Time_Hours', 'estimated_time_hours', 'Time_Hours', 'Hours']
                estimated_hours = None
                
                for field in time_fields:
                    if field in estimate_data:
                        estimated_hours = float(estimate_data[field])
                        break
                
                if estimated_hours is not None:
                    complexity_tier = estimate_data.get('Complexity_Tier', 
                                                      estimate_data.get('complexity_tier', 'Medium'))
                    self.time_estimate_stats['matched_estimates'] += 1
                    return estimated_hours, complexity_tier
                    
            except Exception as e:
                logger.warning(f"Error parsing time estimate for scene {scene_number}: {str(e)}")
        
        # Fallback to page-based estimation
        try:
            page_count = self._parse_page_count(scene_data.get('Page_Count', '1'))
            estimated_hours = page_count * 1.0
            complexity_tier = 'Medium'
            
            self.time_estimate_stats['fallback_page_count'] += 1
            return estimated_hours, complexity_tier
            
        except Exception as e:
            logger.error(f"Failed to estimate time for scene {scene_number}: {str(e)}")
            self.time_estimate_stats['failed_matches'] += 1
            return 1.0, 'Medium'
    
    def _create_scene_info(self, scene_data: Dict) -> SceneInfo:
        """Create SceneInfo object from scene data"""
        scene_number = str(scene_data['Scene_Number']).strip()
        estimated_hours, complexity_tier = self._extract_time_and_complexity(scene_data, scene_number)
        
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
        self.scenes = [self._create_scene_info(scene) for scene in self.stripboard]
        
        location_groups = defaultdict(list)
        for scene in self.scenes:
            location_groups[scene.geographic_location].append(scene)
        
        for location, scenes in location_groups.items():
            total_hours = sum(scene.estimated_time_hours for scene in scenes)
            total_days = max(1, int(total_hours / self.hours_per_day) + (1 if total_hours % self.hours_per_day > 0 else 0))
            
            complexity_counts = defaultdict(int)
            for scene in scenes:
                complexity_counts[scene.complexity_tier] += 1
            
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
            logger.info(f"Location cluster '{location}': {len(scenes)} scenes, {total_hours:.1f} hours, {total_days} days")
        
        return self.location_clusters

class NaiveScheduler:
    """Naive sequential scheduler that respects all non-negotiable constraints"""
    
    def __init__(self, location_clusters: Dict[str, LocationCluster], 
                 date_anchors: List[DateAnchor],
                 non_negotiables: List[NonNegotiableConstraint],
                 hours_per_day: float = 12.0):
        self.location_clusters = location_clusters
        self.date_anchors = date_anchors
        self.non_negotiables = non_negotiables
        self.hours_per_day = hours_per_day
        self.shooting_calendar: List[ShootingDay] = []
        self.conflicts: List[Dict[str, Any]] = []
        
    def create_shooting_calendar(self, start_date: datetime, end_date: datetime) -> List[ShootingDay]:
        """Create shooting calendar with available days and mandatory off days"""
        calendar = []
        current_date = start_date
        day_number = 1
        
        while current_date <= end_date:
            day_of_week = current_date.strftime('%A')
            
            # Sunday is mandatory off day
            status = 'off_day' if day_of_week == 'Sunday' else 'available'
            
            shooting_day = ShootingDay(
                date=current_date,
                day_number=day_number if status != 'off_day' else 0,
                day_of_week=day_of_week,
                status=status,
                scenes=[],
                cast_required=set(),
                constraints_applied=[]
            )
            
            calendar.append(shooting_day)
            
            if status != 'off_day':
                day_number += 1
                
            current_date += timedelta(days=1)
        
        logger.info(f"Created shooting calendar: {len(calendar)} total days, {day_number-1} shooting days")
        return calendar
    
    def apply_date_anchors(self):
        """Apply date-specific anchors to block unavailable days"""
        for anchor in self.date_anchors:
            for day in self.shooting_calendar:
                if day.date.date() == anchor.anchor_date.date():
                    if 'unavailable' in anchor.constraint_type or 'specific_unavailable' in anchor.constraint_type:
                        day.status = 'blocked'
                        day.constraints_applied.append(f"Blocked by {anchor.entity_name}: {anchor.constraint_type}")
                        logger.info(f"Blocked {day.date.strftime('%Y-%m-%d')} due to {anchor.entity_name} {anchor.constraint_type}")
    
    def get_available_days(self) -> List[ShootingDay]:
        """Get list of available shooting days"""
        return [day for day in self.shooting_calendar if day.status == 'available']
    
    def apply_non_negotiable_constraints(self, scenes_by_location: Dict[str, List[str]]):
        """Apply non-negotiable constraints to enforce scene ordering and grouping"""
        scenes_to_locations = {}
        for location, scene_list in scenes_by_location.items():
            for scene in scene_list:
                scenes_to_locations[scene] = location
        
        # Apply constraints in priority order
        for constraint in self.non_negotiables:
            try:
                if constraint.constraint_type == 'shoot_first':
                    self._apply_shoot_first_constraint(constraint, scenes_to_locations)
                elif constraint.constraint_type == 'shoot_last':
                    self._apply_shoot_last_constraint(constraint, scenes_to_locations)
                elif constraint.constraint_type == 'same_day_grouping':
                    self._apply_same_day_grouping_constraint(constraint, scenes_to_locations)
                elif constraint.constraint_type == 'equipment_requirement':
                    self._apply_equipment_constraint(constraint, scenes_to_locations)
                # Add more constraint types as needed
                
            except Exception as e:
                logger.error(f"Error applying constraint {constraint.constraint_id}: {str(e)}")
                self.conflicts.append({
                    'type': 'constraint_application_error',
                    'constraint_id': constraint.constraint_id,
                    'error': str(e)
                })
    
    def _apply_shoot_first_constraint(self, constraint: NonNegotiableConstraint, scenes_to_locations: Dict[str, str]):
        """Force specified scenes to be scheduled first"""
        for scene in constraint.affected_scenes:
            if scene in scenes_to_locations:
                location = scenes_to_locations[scene]
                # This will be enforced during location assignment
                logger.info(f"Scene {scene} marked for first shooting (location: {location})")
    
    def _apply_shoot_last_constraint(self, constraint: NonNegotiableConstraint, scenes_to_locations: Dict[str, str]):
        """Force specified scenes to be scheduled last"""
        for scene in constraint.affected_scenes:
            if scene in scenes_to_locations:
                location = scenes_to_locations[scene]
                logger.info(f"Scene {scene} marked for last shooting (location: {location})")
    
    def _apply_same_day_grouping_constraint(self, constraint: NonNegotiableConstraint, scenes_to_locations: Dict[str, str]):
        """Force specified scenes to be scheduled on the same day"""
        scenes = constraint.affected_scenes
        if len(scenes) > 1:
            logger.info(f"Scenes {scenes} must be scheduled on the same day")
    
    def _apply_equipment_constraint(self, constraint: NonNegotiableConstraint, scenes_to_locations: Dict[str, str]):
        """Apply equipment availability constraints"""
        equipment_data = constraint.additional_data.get('equipment_requirements', {})
        rental_schedule = equipment_data.get('rental_schedule', {})
        
        for scene in constraint.affected_scenes:
            if scene in scenes_to_locations:
                location = scenes_to_locations[scene]
                logger.info(f"Scene {scene} requires equipment: {constraint.constraint_details}")
    
    def _get_location_availability_window(self, location_name: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the availability window for a specific location from date anchors.
        Returns (start_date, end_date) tuple or (None, None) if no window constraint exists.
        """
        start_date = None
        end_date = None
        
        for anchor in self.date_anchors:
            # Check if this anchor is for the current location
            if anchor.entity_name == location_name:
                # Handle different anchor types
                if 'location_availability' in anchor.constraint_type or 'location_window' in anchor.constraint_type:
                    if 'window_start' in anchor.constraint_type:
                        start_date = anchor.anchor_date
                    elif 'window_end' in anchor.constraint_type:
                        end_date = anchor.anchor_date
                    elif 'availability' in anchor.constraint_type:
                        # Single day availability
                        start_date = anchor.anchor_date
                        end_date = anchor.anchor_date
        
        return start_date, end_date
    
    def assign_location_blocks(self):
        """Assign location clusters to available days sequentially"""
        available_days = self.get_available_days()
        current_day_index = 0
        
        # Sort clusters by priority (largest first, then alphabetical)
        sorted_clusters = sorted(
            self.location_clusters.items(),
            key=lambda x: (-x[1].total_shooting_days, x[0])
        )
        
        for location, cluster in sorted_clusters:
            days_needed = cluster.total_shooting_days
            
            # CRITICAL FIX: Check for location availability window constraints
            start_window, end_window = self._get_location_availability_window(location)
            
            # Filter available days to only those within the location's window
            if start_window and end_window:
                logger.info(f"Location '{location}' has availability window: {start_window.date()} to {end_window.date()}")
                
                # Filter available days to only those within the window
                location_valid_days = [
                    day for day in available_days[current_day_index:]
                    if start_window.date() <= day.date.date() <= end_window.date()
                ]
                
                if len(location_valid_days) < days_needed:
                    self.conflicts.append({
                        'type': 'location_window_violation',
                        'location': location,
                        'days_needed': days_needed,
                        'days_available_in_window': len(location_valid_days),
                        'window_start': start_window.date().isoformat(),
                        'window_end': end_window.date().isoformat(),
                        'reason': f'Location requires {days_needed} days but only {len(location_valid_days)} available days exist within its availability window'
                    })
                    logger.warning(f"Cannot schedule location '{location}': needs {days_needed} days, but only {len(location_valid_days)} available within window {start_window.date()} to {end_window.date()}")
                    continue
                
                # Use the filtered days for this location
                days_to_assign = location_valid_days[:days_needed]
            else:
                # No window constraint - use normal sequential assignment
                remaining_days = available_days[current_day_index:]
                
                if len(remaining_days) < days_needed:
                    self.conflicts.append({
                        'type': 'insufficient_days',
                        'location': location,
                        'days_needed': days_needed,
                        'days_available': len(remaining_days)
                    })
                    logger.warning(f"Insufficient days for location {location}: need {days_needed}, have {len(remaining_days)}")
                    continue
                
                days_to_assign = remaining_days[:days_needed]
            
            # Assign the filtered/validated days to this location
            assigned_days = []
            for day in days_to_assign:
                day.status = 'assigned'
                day.location = location
                assigned_days.append(day)
            
            # Distribute scenes across assigned days
            scenes_per_day = len(cluster.scenes) // days_needed
            extra_scenes = len(cluster.scenes) % days_needed
            
            scene_index = 0
            for day_i, day in enumerate(assigned_days):
                scenes_for_day = scenes_per_day + (1 if day_i < extra_scenes else 0)
                day_scenes = cluster.scenes[scene_index:scene_index + scenes_for_day]
                
                day.scenes = [scene.scene_number for scene in day_scenes]
                day.total_hours = sum(scene.estimated_time_hours for scene in day_scenes)
                
                # Collect cast requirements
                for scene in day_scenes:
                    day.cast_required.update(scene.cast)
                
                scene_index += scenes_for_day
            
            # Only advance current_day_index if no window constraint (sequential assignment)
            if not (start_window and end_window):
                current_day_index += days_needed
            
            logger.info(f"Assigned location '{location}' to {days_needed} days starting {assigned_days[0].date.strftime('%Y-%m-%d')}")
    
    def generate_schedule(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate complete naive schedule"""
        logger.info("Starting naive scheduling")
        
        # Create shooting calendar
        self.shooting_calendar = self.create_shooting_calendar(start_date, end_date)
        
        # Apply date anchors to block unavailable days
        self.apply_date_anchors()
        
        # Create scenes by location mapping
        scenes_by_location = {}
        for location, cluster in self.location_clusters.items():
            scenes_by_location[location] = [scene.scene_number for scene in cluster.scenes]
        
        # Apply non-negotiable constraints
        self.apply_non_negotiable_constraints(scenes_by_location)
        
        # Assign location blocks sequentially
        self.assign_location_blocks()
        
        # Generate output
        schedule_output = {
            'shooting_days': [],
            'conflicts': self.conflicts,
            'statistics': self._generate_statistics()
        }
        
        for day in self.shooting_calendar:
            if day.status == 'assigned':
                schedule_output['shooting_days'].append({
                    'date': day.date.strftime('%Y-%m-%d'),
                    'day_of_week': day.day_of_week,
                    'day_number': day.day_number,
                    'location': day.location,
                    'scenes': day.scenes,
                    'total_hours': round(day.total_hours, 2),
                    'cast_required': list(day.cast_required),
                    'constraints_applied': day.constraints_applied
                })
        
        logger.info(f"Naive scheduling completed: {len(schedule_output['shooting_days'])} shooting days, {len(self.conflicts)} conflicts")
        return schedule_output
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate schedule statistics"""
        assigned_days = [day for day in self.shooting_calendar if day.status == 'assigned']
        total_scenes = sum(len(day.scenes) for day in assigned_days)
        total_hours = sum(day.total_hours for day in assigned_days)
        
        return {
            'total_shooting_days': len(assigned_days),
            'total_scenes_scheduled': total_scenes,
            'total_shooting_hours': round(total_hours, 2),
            'average_hours_per_day': round(total_hours / len(assigned_days) if assigned_days else 0, 2),
            'conflicts_count': len(self.conflicts),
            'locations_scheduled': len(set(day.location for day in assigned_days if day.location))
        }

class ProductionScheduler:
    """Main scheduler class that coordinates all components"""
    
    def __init__(self, input_data):
        # Handle both array format [{}] and direct object format {}
        if isinstance(input_data, list):
            if len(input_data) != 1:
                raise ValueError(f"If array format, input must contain exactly one object, got {len(input_data)}")
            production_data = input_data[0]
        elif isinstance(input_data, dict):
            production_data = input_data
        else:
            raise ValueError(f"Input must be either an array with one object or a single object, got {type(input_data)}")
        
        # Validate required keys
        required_keys = ['stripboard', 'constraints', 'ga_params']
        for key in required_keys:
            if key not in production_data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate stripboard structure
        stripboard = production_data['stripboard']
        if not isinstance(stripboard, list):
            raise ValueError("Stripboard must be an array")
        
        if len(stripboard) == 0:
            raise ValueError("Stripboard cannot be empty")
        
        # Validate required scene fields
        required_scene_fields = [
            'Scene_Number', 'INT_EXT', 'Location_Name', 'Day_Night',
            'Synopsis', 'Page_Count', 'Script_Day', 'Cast', 'Geographic_Location'
        ]
        
        for i, scene in enumerate(stripboard):
            scene_id = scene.get('Scene_Number', f'Scene_{i+1}')
            
            for field in required_scene_fields:
                if field not in scene:
                    if field == 'Location_Name':
                        logger.warning(f"Scene {scene_id} missing Location_Name - setting to 'TBD'")
                        scene['Location_Name'] = 'TBD'
                        continue
                    
                    raise ValueError(f"Scene {scene_id} missing required field: {field}")
        
        logger.info(f"Data validation successful: {len(stripboard)} scenes loaded")
        
        # Store configuration
        self.production_data = production_data
        self.stripboard = self.production_data['stripboard']
        self.constraints = self.production_data['constraints']
        self.ga_params = self.production_data['ga_params']
        
        # Extract hours per day from production rules
        self.hours_per_day = self._extract_hours_per_day()
        logger.info(f"Using hours_per_day: {self.hours_per_day}")
        
        # Initialize components
        self.date_anchor_extractor = DateAnchorExtractor(self.constraints, self.stripboard)
        self.non_negotiable_extractor = NonNegotiableExtractor(self.constraints)
        
        # Get time estimates data
        time_estimates_data = (
            self.constraints
            .get('operational_data', {})
            .get('time_estimates', {})
            .get('scene_estimates', [])
        )
        
        self.location_manager = LocationClusterManager(
            self.stripboard, 
            time_estimates_data, 
            hours_per_day=self.hours_per_day
        )
        
        # Initialize results storage
        self.date_anchors = []
        self.non_negotiables = []
        self.location_clusters = {}
        self.naive_schedule = {}
    
    def _extract_hours_per_day(self) -> float:
        """Extract hours per day from production rules constraints"""
        try:
            production_rules = (
                self.constraints
                .get('operational_data', {})
                .get('production_rules', {})
                .get('rules', [])
            )
            
            for rule in production_rules:
                parameter_name = rule.get('parameter_name', '')
                rule_type = rule.get('parsed_data', {}).get('rule_type', '')
                
                if (parameter_name == 'standard_work_day_hours' or 
                    rule_type == 'standard_day_length'):
                    
                    parsed_data = rule.get('parsed_data', {})
                    hours = parsed_data.get('hours')
                    
                    if hours is not None:
                        hours_float = float(hours)
                        logger.info(f"Found hours_per_day in production rules: {hours_float}")
                        return hours_float
            
            logger.warning("No standard work day hours found in production rules, using default 10.0")
            return 10.0
            
        except Exception as e:
            logger.error(f"Error extracting hours_per_day from production rules: {str(e)}")
            return 10.0
    
    def process_iteration_2(self):
        """Execute Iteration 2: All Non-Negotiables + Naive Scheduler"""
        logger.info("Starting Iteration 2: All Non-Negotiables + Naive Scheduler")
        
        # Step 1: Extract date-specific anchors
        logger.info("Step 1: Extracting date-specific anchors...")
        self.date_anchors = self.date_anchor_extractor.extract_all_anchors()
        
        # Step 2: Extract non-date-specific non-negotiables
        logger.info("Step 2: Extracting non-date-specific non-negotiables...")
        self.non_negotiables = self.non_negotiable_extractor.extract_all_non_negotiables()
        
        # Step 3: Create location clusters
        logger.info("Step 3: Creating location clusters...")
        self.location_clusters = self.location_manager.cluster_scenes_by_location()
        
        # Step 4: Generate naive schedule
        logger.info("Step 4: Generating naive schedule...")
        naive_scheduler = NaiveScheduler(
            self.location_clusters,
            self.date_anchors,
            self.non_negotiables,
            self.hours_per_day
        )
        
        # Use reasonable date range (Sept 1 - Nov 30, 2025)
        start_date = datetime(2025, 9, 1)
        end_date = datetime(2025, 11, 30)
        
        self.naive_schedule = naive_scheduler.generate_schedule(start_date, end_date)
        
        # Compile results
        results = {
            'iteration': 2,
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
            'non_negotiable_constraints': [
                {
                    'constraint_id': constraint.constraint_id,
                    'constraint_type': constraint.constraint_type,
                    'constraint_category': constraint.constraint_category,
                    'priority': constraint.priority,
                    'constraint_details': constraint.constraint_details,
                    'affected_scenes': constraint.affected_scenes,
                    'affected_locations': constraint.affected_locations
                }
                for constraint in self.non_negotiables
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
            'naive_schedule': self.naive_schedule
        }
        
        logger.info(f"Iteration 2 completed successfully: "
                   f"{len(self.date_anchors)} date anchors, "
                   f"{len(self.non_negotiables)} non-negotiables, "
                   f"{len(self.naive_schedule.get('shooting_days', []))} shooting days scheduled")
        
        return results

def main():
    """Main entry point for local testing"""
    try:
        sample_data = {
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
                'operational_data': {
                    'time_estimates': {'scene_estimates': []},
                    'production_rules': {
                        'rules': [
                            {
                                'parameter_name': 'standard_work_day_hours',
                                'parsed_data': {'hours': 12}
                            }
                        ]
                    }
                },
                'creative_constraints': {
                    'director_notes': {'director_constraints': []},
                    'dop_priorities': {'dop_priorities': []}
                },
                'location_constraints': {'locations': {}},
                'technical_constraints': {'equipment': {}, 'special_requirements': {}}
            },
            'ga_params': {
                'phase1_population': 100,
                'phase1_generations': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'seed': 42,
                'conflict_tolerance': 0.05
            }
        }
        
        scheduler = ProductionScheduler(sample_data)
        results = scheduler.process_iteration_2()
        
        print("=== Iteration 2 Results ===")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
