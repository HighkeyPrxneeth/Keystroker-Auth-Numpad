from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class AuthenticationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"


class KeystrokeEvent(BaseModel):
    """Single keystroke event with timing"""
    key: str
    key_code: int
    timestamp: float
    event_type: str  # 'keydown' or 'keyup'


class KeystrokePattern(BaseModel):
    """Processed keystroke pattern with extracted features"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    text_typed: str
    raw_events: List[KeystrokeEvent]
    
    # Timing features
    hold_times: List[float] = Field(default_factory=list)
    down_down_times: List[float] = Field(default_factory=list)
    up_down_times: List[float] = Field(default_factory=list)
    
    # Statistical features
    total_typing_time: float
    typing_speed: float  # characters per second
    rhythm_variance: float
    
    # Feature vector for ML
    feature_vector: List[float] = Field(default_factory=list)
    feature_names: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class User(BaseModel):
    """User with keystroke authentication profile"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    
    # Authentication patterns
    enrollment_patterns: List[str] = Field(default_factory=list)  # Pattern IDs
    is_enrolled: bool = False
    enrollment_completed_at: Optional[datetime] = None
    
    # Model performance metrics
    authentication_accuracy: Optional[float] = None
    false_acceptance_rate: Optional[float] = None
    false_rejection_rate: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AuthenticationAttempt(BaseModel):
    """Authentication attempt record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    pattern_id: str
    
    status: AuthenticationStatus
    confidence_score: float
    processing_time_ms: float
    
    # Model predictions
    model_predictions: Dict[str, float] = Field(default_factory=dict)  # {model_name: confidence}
    ensemble_score: float
    threshold_used: float
    
    # Metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Request/Response Models
class KeystrokeDataInput(BaseModel):
    """Input for keystroke data collection"""
    text_typed: str
    events: List[KeystrokeEvent]


class EnrollmentRequest(BaseModel):
    """Request to enroll a user with keystroke patterns"""
    username: str
    email: str
    patterns: List[KeystrokeDataInput]


class AuthenticationRequest(BaseModel):
    """Request to authenticate user based on keystroke pattern"""
    username: str
    pattern: KeystrokeDataInput


class EnrollmentResponse(BaseModel):
    """Response after user enrollment"""
    user_id: str
    status: str
    patterns_processed: int
    message: str


class AuthenticationResponse(BaseModel):
    """Response after authentication attempt"""
    user_id: str
    status: AuthenticationStatus
    confidence_score: float
    processing_time_ms: float
    message: str


class UserCreate(BaseModel):
    """Request to create a new user"""
    username: str
    email: str


class UserResponse(BaseModel):
    """User information response"""
    id: str
    username: str
    email: str
    is_enrolled: bool
    enrollment_patterns_count: int
    authentication_accuracy: Optional[float]
    created_at: datetime