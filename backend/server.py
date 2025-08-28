from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our custom modules
from models import *
from feature_extraction import KeystrokeFeatureExtractor, validate_keystroke_data
from ml_pipeline import KeystrokeMLPipeline

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize ML components
feature_extractor = KeystrokeFeatureExtractor()
ml_pipeline = KeystrokeMLPipeline()

# Thread pool for ML operations
executor = ThreadPoolExecutor(max_workers=4)

# Create the main app without a prefix
app = FastAPI(
    title="Keystroke Dynamics Authentication API",
    description="Behavioral biometric authentication using keystroke timing patterns",
    version="1.0.0"
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Database collections
users_collection = db.users
patterns_collection = db.keystroke_patterns
attempts_collection = db.authentication_attempts

# Basic health check routes
@api_router.get("/")
async def root():
    return {"message": "Keystroke Dynamics Authentication API", "status": "active"}

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "ml_pipeline_status": "active",
        "database_status": "connected"
    }

# User management routes
@api_router.post("/users", response_model=UserResponse)
async def create_user(user_data: UserCreate):
    """Create a new user for keystroke authentication"""
    
    # Check if user already exists
    existing_user = await users_collection.find_one({"username": user_data.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    existing_email = await users_collection.find_one({"email": user_data.email})
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Create new user
    user = User(
        username=user_data.username,
        email=user_data.email
    )
    
    await users_collection.insert_one(user.dict())
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_enrolled=user.is_enrolled,
        enrollment_patterns_count=len(user.enrollment_patterns),
        authentication_accuracy=user.authentication_accuracy,
        created_at=user.created_at
    )

@api_router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get user information"""
    
    user_data = await users_collection.find_one({"id": user_id})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = User(**user_data)
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_enrolled=user.is_enrolled,
        enrollment_patterns_count=len(user.enrollment_patterns),
        authentication_accuracy=user.authentication_accuracy,
        created_at=user.created_at
    )

@api_router.get("/users", response_model=List[UserResponse])
async def list_users(skip: int = 0, limit: int = 100):
    """List all users"""
    
    cursor = users_collection.find().skip(skip).limit(limit)
    users_data = await cursor.to_list(length=limit)
    
    users = []
    for user_data in users_data:
        user = User(**user_data)
        users.append(UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            is_enrolled=user.is_enrolled,
            enrollment_patterns_count=len(user.enrollment_patterns),
            authentication_accuracy=user.authentication_accuracy,
            created_at=user.created_at
        ))
    
    return users

# Keystroke pattern collection and processing
@api_router.post("/collect-pattern")
async def collect_keystroke_pattern(
    user_id: str,
    pattern_data: KeystrokeDataInput,
    background_tasks: BackgroundTasks
):
    """Collect and process keystroke pattern for a user"""
    
    # Validate user exists
    user_data = await users_collection.find_one({"id": user_id})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Validate keystroke data
    if not validate_keystroke_data(pattern_data.events):
        raise HTTPException(
            status_code=400, 
            detail="Insufficient keystroke data. Need at least 10 events with both keydown/keyup."
        )
    
    # Extract features in background
    loop = asyncio.get_event_loop()
    
    try:
        # Extract features using thread pool
        pattern = await loop.run_in_executor(
            executor,
            feature_extractor.extract_features,
            pattern_data.events,
            pattern_data.text_typed
        )
        
        # Set user ID
        pattern.user_id = user_id
        
        # Store pattern in database
        await patterns_collection.insert_one(pattern.dict())
        
        return {
            "pattern_id": pattern.id,
            "user_id": user_id,
            "features_extracted": len(pattern.feature_vector),
            "processing_time": pattern.total_typing_time,
            "typing_speed": pattern.typing_speed,
            "status": "processed"
        }
        
    except Exception as e:
        logger.error(f"Error processing keystroke pattern: {e}")
        raise HTTPException(status_code=500, detail="Error processing keystroke pattern")

# User enrollment
@api_router.post("/enroll", response_model=EnrollmentResponse)
async def enroll_user(enrollment_data: EnrollmentRequest):
    """Enroll user with multiple keystroke patterns for training"""
    
    if len(enrollment_data.patterns) < 5:
        raise HTTPException(
            status_code=400,
            detail="Need at least 5 keystroke patterns for enrollment"
        )
    
    try:
        # Create user if doesn't exist
        existing_user = await users_collection.find_one({"username": enrollment_data.username})
        if existing_user:
            user = User(**existing_user)
        else:
            user = User(
                username=enrollment_data.username,
                email=enrollment_data.email
            )
            await users_collection.insert_one(user.dict())
        
        # Process all patterns
        feature_vectors = []
        pattern_ids = []
        
        loop = asyncio.get_event_loop()
        
        for pattern_data in enrollment_data.patterns:
            # Validate pattern data
            if not validate_keystroke_data(pattern_data.events):
                continue
            
            # Extract features
            pattern = await loop.run_in_executor(
                executor,
                feature_extractor.extract_features,
                pattern_data.events,
                pattern_data.text_typed
            )
            pattern.user_id = user.id
            
            # Store pattern
            await patterns_collection.insert_one(pattern.dict())
            
            feature_vectors.append(pattern.feature_vector)
            pattern_ids.append(pattern.id)
        
        if len(feature_vectors) < 3:
            raise HTTPException(
                status_code=400,
                detail="Insufficient valid patterns for enrollment. Need at least 3 valid patterns."
            )
        
        # Generate negative samples (imposter data) by using other users' patterns
        imposter_patterns = await patterns_collection.find(
            {"user_id": {"$ne": user.id}}
        ).limit(len(feature_vectors) * 2).to_list(None)
        
        # Prepare training data
        all_features = feature_vectors.copy()
        labels = [user.id] * len(feature_vectors)  # Positive samples
        
        # Add imposter samples
        for imp_pattern_data in imposter_patterns:
            imp_pattern = KeystrokePattern(**imp_pattern_data)
            all_features.append(imp_pattern.feature_vector)
            labels.append("imposter")
        
        # Train ML model
        training_result = await loop.run_in_executor(
            executor,
            ml_pipeline.train_user_model,
            user.id,
            all_features,
            labels
        )
        
        # Update user enrollment status
        user.enrollment_patterns = pattern_ids
        user.is_enrolled = True
        user.enrollment_completed_at = datetime.utcnow()
        
        await users_collection.update_one(
            {"id": user.id},
            {"$set": user.dict()}
        )
        
        return EnrollmentResponse(
            user_id=user.id,
            status="enrolled",
            patterns_processed=len(pattern_ids),
            message=f"Successfully enrolled with {len(pattern_ids)} patterns. Model accuracy: {training_result['model_scores'].get('ensemble', 0):.2f}"
        )
        
    except Exception as e:
        logger.error(f"Error during user enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")

# Authentication
@api_router.post("/authenticate", response_model=AuthenticationResponse)
async def authenticate_user(auth_request: AuthenticationRequest):
    """Authenticate user based on keystroke pattern"""
    
    start_time = datetime.now()
    
    try:
        # Find user
        user_data = await users_collection.find_one({"username": auth_request.username})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = User(**user_data)
        
        if not user.is_enrolled:
            raise HTTPException(status_code=400, detail="User not enrolled")
        
        # Validate keystroke data
        if not validate_keystroke_data(auth_request.pattern.events):
            return AuthenticationResponse(
                user_id=user.id,
                status=AuthenticationStatus.INSUFFICIENT_DATA,
                confidence_score=0.0,
                processing_time_ms=0.0,
                message="Insufficient keystroke data for authentication"
            )
        
        # Extract features from auth pattern
        loop = asyncio.get_event_loop()
        
        auth_pattern = await loop.run_in_executor(
            executor,
            feature_extractor.extract_features,
            auth_request.pattern.events,
            auth_request.pattern.text_typed
        )
        
        # Authenticate using ML pipeline
        auth_result = await loop.run_in_executor(
            executor,
            ml_pipeline.authenticate_user,
            user.id,
            auth_pattern.feature_vector
        )
        
        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create authentication attempt record
        attempt = AuthenticationAttempt(
            user_id=user.id,
            pattern_id=auth_pattern.id,
            status=AuthenticationStatus.SUCCESS if auth_result['authenticated'] else AuthenticationStatus.FAILED,
            confidence_score=auth_result['confidence'],
            processing_time_ms=processing_time,
            model_predictions=auth_result['model_predictions'],
            ensemble_score=auth_result['confidence'],
            threshold_used=auth_result['threshold_used']
        )
        
        # Store attempt
        await attempts_collection.insert_one(attempt.dict())
        
        return AuthenticationResponse(
            user_id=user.id,
            status=attempt.status,
            confidence_score=auth_result['confidence'],
            processing_time_ms=processing_time,
            message=auth_result['reason']
        )
        
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

# Analytics and monitoring
@api_router.get("/users/{user_id}/performance")
async def get_user_performance(user_id: str):
    """Get authentication performance metrics for a user"""
    
    user_data = await users_collection.find_one({"id": user_id})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get recent authentication attempts
    attempts = await attempts_collection.find(
        {"user_id": user_id}
    ).sort("created_at", -1).limit(100).to_list(None)
    
    if not attempts:
        return {"message": "No authentication attempts found", "user_id": user_id}
    
    # Calculate metrics
    total_attempts = len(attempts)
    successful_attempts = sum(1 for a in attempts if a['status'] == 'success')
    failed_attempts = total_attempts - successful_attempts
    
    avg_confidence = sum(a['confidence_score'] for a in attempts) / total_attempts
    avg_processing_time = sum(a['processing_time_ms'] for a in attempts) / total_attempts
    
    # Get model performance from ML pipeline
    model_perf = ml_pipeline.get_model_performance(user_id)
    
    return {
        "user_id": user_id,
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "failed_attempts": failed_attempts,
        "success_rate": successful_attempts / total_attempts if total_attempts > 0 else 0,
        "average_confidence": avg_confidence,
        "average_processing_time_ms": avg_processing_time,
        "model_performance": model_perf,
        "last_attempt": attempts[0]['created_at'] if attempts else None
    }

@api_router.get("/system/stats")
async def get_system_stats():
    """Get overall system statistics"""
    
    total_users = await users_collection.count_documents({})
    enrolled_users = await users_collection.count_documents({"is_enrolled": True})
    total_patterns = await patterns_collection.count_documents({})
    total_attempts = await attempts_collection.count_documents({})
    
    # Recent authentication success rate
    recent_attempts = await attempts_collection.find().sort("created_at", -1).limit(1000).to_list(None)
    
    if recent_attempts:
        recent_success_rate = sum(1 for a in recent_attempts if a['status'] == 'success') / len(recent_attempts)
        avg_processing_time = sum(a['processing_time_ms'] for a in recent_attempts) / len(recent_attempts)
    else:
        recent_success_rate = 0
        avg_processing_time = 0
    
    return {
        "total_users": total_users,
        "enrolled_users": enrolled_users,
        "total_patterns": total_patterns,
        "total_attempts": total_attempts,
        "recent_success_rate": recent_success_rate,
        "average_processing_time_ms": avg_processing_time,
        "system_status": "operational"
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
