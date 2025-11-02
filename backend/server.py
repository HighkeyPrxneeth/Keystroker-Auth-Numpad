import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from feature_extraction import KeystrokeFeatureExtractor, validate_keystroke_data
from ml_pipeline import KeystrokeMLPipeline

# Import our custom modules
from models import (
    AuthenticationAttempt,
    AuthenticationRequest,
    AuthenticationResponse,
    AuthenticationStatus,
    EnrollmentRequest,
    EnrollmentResponse,
    InputDeviceType,
    KeystrokeDataInput,
    KeystrokePattern,
    User,
    UserCreate,
    UserResponse,
)
from motor.motor_asyncio import AsyncIOMotorClient
from starlette.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# MongoDB connection
mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

# Initialize ML components
feature_extractor = KeystrokeFeatureExtractor()
ml_pipeline = KeystrokeMLPipeline()

# Thread pool for ML operations
executor = ThreadPoolExecutor(max_workers=4)
device_type_training_lock = asyncio.Lock()

# Create the main app without a prefix
app = FastAPI(
    title="Keystroke Dynamics Authentication API",
    description="Behavioral biometric authentication using keystroke timing patterns",
    version="1.0.0",
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Database collections
users_collection = db.users
patterns_collection = db.keystroke_patterns
attempts_collection = db.authentication_attempts


async def retrain_device_type_classifier():
    """Retrain the global classifier that differentiates numpad vs key-row input."""

    async with device_type_training_lock:
        cursor = patterns_collection.find(
            {
                "input_device_type": {
                    "$in": [
                        InputDeviceType.KEYROW.value,
                        InputDeviceType.NUMPAD.value,
                    ]
                }
            }
        )
        pattern_docs = await cursor.to_list(length=None)

        feature_vectors = []
        labels = []

        for doc in pattern_docs:
            feature_vector = doc.get("feature_vector")
            device_type = doc.get("input_device_type")
            if feature_vector and device_type:
                feature_vectors.append(feature_vector)
                labels.append(device_type)

        if len(feature_vectors) < 4 or len(set(labels)) < 2:
            logger.info(
                "Skipping device type training (samples=%d, types=%d)",
                len(feature_vectors),
                len(set(labels)),
            )
            return None

        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                executor,
                ml_pipeline.train_device_type_model,
                feature_vectors,
                labels,
            )
            logger.info(
                "Device type model updated. Scores: %s",
                result.get("model_scores", {}),
            )
            return result
        except Exception as exc:
            logger.error(f"Device type training failed: {exc}")
            return None


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
        "database_status": "connected",
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
    user = User(username=user_data.username, email=user_data.email)

    await users_collection.insert_one(user.dict())

    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_enrolled=user.is_enrolled,
        enrollment_patterns_count=len(user.enrollment_patterns),
        authentication_accuracy=user.authentication_accuracy,
        created_at=user.created_at,
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
        created_at=user.created_at,
    )


@api_router.get("/users", response_model=List[UserResponse])
async def list_users(skip: int = 0, limit: int = 100):
    """List all users"""

    cursor = users_collection.find().skip(skip).limit(limit)
    users_data = await cursor.to_list(length=limit)

    users = []
    for user_data in users_data:
        user = User(**user_data)
        users.append(
            UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                is_enrolled=user.is_enrolled,
                enrollment_patterns_count=len(user.enrollment_patterns),
                authentication_accuracy=user.authentication_accuracy,
                created_at=user.created_at,
            )
        )


@api_router.post("/classify-device", response_model=DeviceClassificationResponse)
async def classify_input_device(pattern: KeystrokeDataInput):
    """Classify whether a keystroke pattern originated from numpad or key row."""

    if not validate_keystroke_data(pattern.events):
        raise HTTPException(
            status_code=400,
            detail="Insufficient keystroke data for classification",
        )

    loop = asyncio.get_event_loop()

    extracted_pattern = await loop.run_in_executor(
        executor,
        feature_extractor.extract_features,
        pattern.events,
        pattern.text_typed,
        pattern.input_device_type,
    )

    classification = await loop.run_in_executor(
        executor,
        ml_pipeline.classify_device_type,
        extracted_pattern.feature_vector,
    )

    if not classification:
        raise HTTPException(
            status_code=400,
            detail="Device type model is not trained yet",
        )

    predicted_label = classification.get("predicted_device_type")
    confidence = classification.get("confidence", 0.0)
    model_used = classification.get("model_used", "ensemble")
    probabilities = classification.get("probabilities", {})

    try:
        predicted_device = InputDeviceType(predicted_label)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=500,
            detail="Device classifier returned an unknown label",
        )

    return DeviceClassificationResponse(
        predicted_device_type=predicted_device,
        confidence=float(confidence),
        model_used=model_used,
        probabilities={str(key): float(value) for key, value in probabilities.items()},
    )

    return users


# Keystroke pattern collection and processing
@api_router.post("/collect-pattern")
async def collect_keystroke_pattern(
    user_id: str, pattern_data: KeystrokeDataInput, background_tasks: BackgroundTasks
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
            detail="Insufficient keystroke data. Need at least 10 events with both keydown/keyup.",
        )

    # Extract features in background
    loop = asyncio.get_event_loop()

    try:
        # Extract features using thread pool
        pattern = await loop.run_in_executor(
            executor,
            feature_extractor.extract_features,
            pattern_data.events,
            pattern_data.text_typed,
            pattern_data.input_device_type,
        )

        # Set user ID
        pattern.user_id = user_id

        # Store pattern in database
        await patterns_collection.insert_one(pattern.dict())

        if pattern.input_device_type:
            asyncio.create_task(retrain_device_type_classifier())

        return {
            "pattern_id": pattern.id,
            "user_id": user_id,
            "features_extracted": len(pattern.feature_vector),
            "processing_time": pattern.total_typing_time,
            "typing_speed": pattern.typing_speed,
            "status": "processed",
        }

    except Exception as e:
        logger.error(f"Error processing keystroke pattern: {e}")
        raise HTTPException(
            status_code=500, detail="Error processing keystroke pattern"
        )


# User enrollment
@api_router.post("/enroll", response_model=EnrollmentResponse)
async def enroll_user(enrollment_data: EnrollmentRequest):
    """Enroll user with multiple keystroke patterns for training"""

    if len(enrollment_data.patterns) < 5:
        raise HTTPException(
            status_code=400, detail="Need at least 5 keystroke patterns for enrollment"
        )

    try:
        # Create user if doesn't exist
        existing_user = await users_collection.find_one(
            {"username": enrollment_data.username}
        )
        if existing_user:
            user = User(**existing_user)
        else:
            user = User(username=enrollment_data.username, email=enrollment_data.email)
            await users_collection.insert_one(user.dict())

        # Process all patterns
        feature_vectors = []
        pattern_ids = []
        device_labeled = False

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
                pattern_data.text_typed,
                pattern_data.input_device_type,
            )
            pattern.user_id = user.id

            # Store pattern
            await patterns_collection.insert_one(pattern.dict())

            feature_vectors.append(pattern.feature_vector)
            pattern_ids.append(pattern.id)

            if pattern.input_device_type:
                device_labeled = True

        if len(feature_vectors) < 3:
            raise HTTPException(
                status_code=400,
                detail="Insufficient valid patterns for enrollment. Need at least 3 valid patterns.",
            )

        # Generate negative samples (imposter data) by using other users' patterns
        imposter_patterns = (
            await patterns_collection.find({"user_id": {"$ne": user.id}})
            .limit(len(feature_vectors) * 2)
            .to_list(None)
        )

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
            executor, ml_pipeline.train_user_model, user.id, all_features, labels
        )

        if device_labeled:
            asyncio.create_task(retrain_device_type_classifier())

        # Update user enrollment status
        user.enrollment_patterns = pattern_ids
        user.is_enrolled = True
        user.enrollment_completed_at = datetime.utcnow()

        await users_collection.update_one({"id": user.id}, {"$set": user.dict()})

        return EnrollmentResponse(
            user_id=user.id,
            status="enrolled",
            patterns_processed=len(pattern_ids),
            message=f"Successfully enrolled with {len(pattern_ids)} patterns. Model accuracy: {training_result['model_scores'].get('ensemble', 0):.2f}",
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
                message="Insufficient keystroke data for authentication",
            )

        # Extract features from auth pattern
        loop = asyncio.get_event_loop()

        auth_pattern = await loop.run_in_executor(
            executor,
            feature_extractor.extract_features,
            auth_request.pattern.events,
            auth_request.pattern.text_typed,
            auth_request.pattern.input_device_type,
        )

        if auth_request.pattern.input_device_type:
            auth_pattern.user_id = user.id
            await patterns_collection.insert_one(auth_pattern.dict())
            asyncio.create_task(retrain_device_type_classifier())

        # Authenticate using ML pipeline
        auth_result = await loop.run_in_executor(
            executor,
            ml_pipeline.authenticate_user,
            user.id,
            auth_pattern.feature_vector,
        )

        device_classification = await loop.run_in_executor(
            executor,
            ml_pipeline.classify_device_type,
            auth_pattern.feature_vector,
        )

        predicted_device_type = None
        device_type_confidence = None
        device_type_probabilities = None

        if device_classification:
            predicted_label = device_classification.get("predicted_device_type")
            if predicted_label is not None:
                try:
                    predicted_device_type = InputDeviceType(predicted_label)
                except ValueError:
                    predicted_device_type = None

            device_type_confidence = device_classification.get("confidence")
            raw_probabilities = device_classification.get("probabilities") or {}
            device_type_probabilities = {
                str(key): float(value) for key, value in raw_probabilities.items()
            }

        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Create authentication attempt record
        attempt = AuthenticationAttempt(
            user_id=user.id,
            pattern_id=auth_pattern.id,
            status=AuthenticationStatus.SUCCESS
            if auth_result["authenticated"]
            else AuthenticationStatus.FAILED,
            confidence_score=auth_result["confidence"],
            processing_time_ms=processing_time,
            model_predictions=auth_result["model_predictions"],
            ensemble_score=auth_result["confidence"],
            threshold_used=auth_result["threshold_used"],
            predicted_device_type=predicted_device_type,
            reported_device_type=auth_request.pattern.input_device_type,
        )

        # Store attempt
        await attempts_collection.insert_one(attempt.dict())

        return AuthenticationResponse(
            user_id=user.id,
            status=attempt.status,
            confidence_score=auth_result["confidence"],
            processing_time_ms=processing_time,
            message=auth_result["reason"],
            predicted_device_type=predicted_device_type,
            device_type_confidence=device_type_confidence,
            device_type_probabilities=device_type_probabilities,
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
    attempts = (
        await attempts_collection.find({"user_id": user_id})
        .sort("created_at", -1)
        .limit(100)
        .to_list(None)
    )

    if not attempts:
        return {
            "user_id": user_id,
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
            "success_rate": 0.0,
            "average_confidence": 0.0,
            "average_processing_time_ms": 0.0,
            "model_performance": ml_pipeline.get_model_performance(user_id),
            "last_attempt": None,
            "message": "No authentication attempts found",
        }

    # Calculate metrics
    total_attempts = len(attempts)
    successful_attempts = sum(
        1 for a in attempts if a.get("status", "").lower() == "success"
    )
    failed_attempts = total_attempts - successful_attempts

    avg_confidence = (
        sum(a.get("confidence_score", 0) for a in attempts) / total_attempts
    )
    avg_processing_time = (
        sum(a.get("processing_time_ms", 0) for a in attempts) / total_attempts
    )

    # Get model performance from ML pipeline
    model_perf = ml_pipeline.get_model_performance(user_id)

    return {
        "user_id": user_id,
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "failed_attempts": failed_attempts,
        "success_rate": successful_attempts / total_attempts
        if total_attempts > 0
        else 0,
        "average_confidence": avg_confidence,
        "average_processing_time_ms": avg_processing_time,
        "model_performance": model_perf,
        "last_attempt": attempts[0]["created_at"] if attempts else None,
    }


@api_router.get("/system/stats")
async def get_system_stats():
    """Get overall system statistics"""

    total_users = await users_collection.count_documents({})
    enrolled_users = await users_collection.count_documents({"is_enrolled": True})
    total_patterns = await patterns_collection.count_documents({})
    total_attempts = await attempts_collection.count_documents({})

    # Recent authentication success rate
    recent_attempts = (
        await attempts_collection.find()
        .sort("created_at", -1)
        .limit(1000)
        .to_list(None)
    )

    if recent_attempts:
        recent_success_rate = sum(
            1 for a in recent_attempts if a["status"] == "success"
        ) / len(recent_attempts)
        avg_processing_time = sum(
            a["processing_time_ms"] for a in recent_attempts
        ) / len(recent_attempts)
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
        "system_status": "operational",
    }


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
