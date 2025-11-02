import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class KeystrokeMLPipeline:
    """Machine learning pipeline for keystroke dynamics authentication"""

    def __init__(self, model_save_path: str = "./models/"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)

        # Initialize models
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.smote = SMOTE(
            random_state=42, sampling_strategy="auto", k_neighbors=3
        )  # Reduced k_neighbors for small datasets

        # Performance metrics
        self.model_performance = {}
        self.ensemble_model = None

        # Device type classification assets
        self.device_type_models = {}
        self.device_type_scaler = StandardScaler()
        self.device_type_label_encoder = LabelEncoder()
        self.device_type_model_performance = {}
        self.device_type_model_path = self.model_save_path / "device_type_models.joblib"
        self.device_type_ensemble = None

        # Authentication thresholds
        self.auth_thresholds = {}
        self.default_threshold = 0.7

        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models with optimized hyperparameters"""

        # Support Vector Machine with RBF kernel
        self.models["svm"] = SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42
        )

        # Random Forest
        self.models["random_forest"] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        # K-Nearest Neighbors - use conservative default
        self.models["knn"] = KNeighborsClassifier(
            n_neighbors=3,  # Reduced from 5 to be safer with small datasets
            weights="distance",
            metric="euclidean",
        )

        # Gradient Boosting
        self.models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )

        logger.info("Initialized ML models: %s", list(self.models.keys()))

    def train_user_model(
        self, user_id: str, feature_vectors: List[List[float]], labels: List[str]
    ) -> Dict[str, Any]:
        """
        Train personalized authentication model for a user

        Args:
            user_id: Unique user identifier
            feature_vectors: List of feature vectors from keystroke patterns
            labels: List of labels (user_id for positive samples, 'imposter' for negative)

        Returns:
            Training results and model performance metrics
        """

        if len(feature_vectors) < 5:
            raise ValueError(
                f"Insufficient training data. Need at least 5 samples, got {len(feature_vectors)}"
            )

        logger.info(
            f"Training model for user {user_id} with {len(feature_vectors)} samples"
        )

        # Convert to numpy arrays
        X = np.array(feature_vectors)

        # Encode labels (user_id = 1, imposter = 0)
        y_binary = np.array([1 if label == user_id else 0 for label in labels])

        # Apply SMOTE for data balancing with adaptive k_neighbors
        n_samples = len(X)

        # Skip SMOTE for very small datasets to avoid neighbor issues
        if n_samples < 10:
            logger.info(f"Skipping SMOTE for small dataset ({n_samples} samples)")
            X_resampled, y_resampled = X, y_binary
        else:
            k_neighbors = min(3, max(1, n_samples - 1))  # Ensure k_neighbors is valid

            try:
                smote = SMOTE(
                    random_state=42, sampling_strategy="auto", k_neighbors=k_neighbors
                )
                X_resampled, y_resampled = smote.fit_resample(X, y_binary)
                logger.info(
                    f"After SMOTE with k_neighbors={k_neighbors}: {len(X_resampled)} samples (original: {len(X)})"
                )
            except Exception as e:
                logger.warning(
                    f"SMOTE failed with k_neighbors={k_neighbors}: {e}, using original data"
                )
                X_resampled, y_resampled = X, y_binary

        # Scale features
        X_scaled = self.scaler.fit_transform(X_resampled)

        # Train individual models
        model_scores = {}
        trained_models = {}

        # Adjust CV folds based on available data
        n_folds = min(5, len(X_resampled) // 2)
        if n_folds < 2:
            n_folds = 2  # Minimum for cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        logger.info(
            f"Using {n_folds}-fold cross-validation for {len(X_resampled)} samples"
        )

        for model_name, model in self.models.items():
            try:
                # Clone and train model
                model_clone = model.__class__(**model.get_params())

                # Special handling for KNN to ensure n_neighbors is valid
                if model_name == "knn":
                    n_neighbors = model_clone.get_params()["n_neighbors"]
                    max_neighbors = len(X_scaled) - 1
                    if n_neighbors > max_neighbors:
                        model_clone.set_params(n_neighbors=max(1, max_neighbors))
                        logger.info(
                            f"Adjusted KNN n_neighbors from {n_neighbors} to {max(1, max_neighbors)}"
                        )

                model_clone.fit(X_scaled, y_resampled)

                # Cross-validation score
                cv_scores = cross_val_score(
                    model_clone, X_scaled, y_resampled, cv=cv, scoring="roc_auc"
                )

                model_scores[model_name] = {
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "model": model_clone,
                }

                trained_models[model_name] = model_clone

                logger.info(
                    f"{model_name} CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
                )

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue

        # Create ensemble model
        ensemble_estimators = [
            (name, info["model"]) for name, info in model_scores.items()
        ]

        if len(ensemble_estimators) > 0:
            self.ensemble_model = VotingClassifier(
                estimators=ensemble_estimators, voting="soft"
            )
            self.ensemble_model.fit(X_scaled, y_resampled)

            # Evaluate ensemble
            ensemble_cv_scores = cross_val_score(
                self.ensemble_model, X_scaled, y_resampled, cv=cv, scoring="roc_auc"
            )

            model_scores["ensemble"] = {
                "cv_mean": ensemble_cv_scores.mean(),
                "cv_std": ensemble_cv_scores.std(),
                "model": self.ensemble_model,
            }

            logger.info(
                f"Ensemble CV AUC: {ensemble_cv_scores.mean():.3f} (+/- {ensemble_cv_scores.std() * 2:.3f})"
            )

        # Save models and preprocessing objects
        self._save_user_model(user_id, trained_models, self.scaler)

        # Determine optimal threshold
        threshold = self._calculate_optimal_threshold(X_scaled, y_resampled)
        self.auth_thresholds[user_id] = threshold

        # Store performance metrics
        self.model_performance[user_id] = model_scores

        return {
            "user_id": user_id,
            "samples_trained": len(X_resampled),
            "original_samples": len(X),
            "model_scores": {
                name: info["cv_mean"] for name, info in model_scores.items()
            },
            "optimal_threshold": threshold,
            "training_completed_at": datetime.utcnow().isoformat(),
        }

    def authenticate_user(
        self, user_id: str, feature_vector: List[float]
    ) -> Dict[str, Any]:
        """
        Authenticate user based on keystroke pattern

        Args:
            user_id: User to authenticate
            feature_vector: Feature vector from current keystroke pattern

        Returns:
            Authentication result with confidence scores
        """

        start_time = datetime.now()

        # Load user model if not in memory
        user_models = self._load_user_model(user_id)
        if not user_models:
            return {
                "authenticated": False,
                "confidence": 0.0,
                "reason": "User model not found",
                "processing_time_ms": 0,
            }

        # Prepare input
        X = np.array([feature_vector])
        X_scaled = user_models["scaler"].transform(X)

        # Get predictions from all models
        predictions = {}
        confidence_scores = {}

        for model_name, model in user_models["models"].items():
            try:
                # Get probability prediction
                proba = model.predict_proba(X_scaled)[0]

                # Handle both binary and single-class scenarios
                # Binary: [imposter_prob, user_prob] -> use proba[1]
                # Single class: [user_prob] -> use proba[0]
                if len(proba) > 1:
                    # Binary classification - get positive class probability
                    user_confidence = proba[1]
                else:
                    # Single class trained - if model only saw positive samples
                    # The single probability represents confidence in that class
                    user_confidence = proba[0]

                confidence_scores[model_name] = float(user_confidence)
                predictions[model_name] = user_confidence

            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {e}")
                continue

        # Ensemble prediction
        ensemble_confidence = 0.0
        if "ensemble" in user_models["models"]:
            try:
                ensemble_proba = user_models["models"]["ensemble"].predict_proba(
                    X_scaled
                )[0]
                # Handle both binary and single-class scenarios
                if len(ensemble_proba) > 1:
                    ensemble_confidence = float(ensemble_proba[1])
                else:
                    ensemble_confidence = float(ensemble_proba[0])
                confidence_scores["ensemble"] = ensemble_confidence
            except Exception as e:
                logger.error(f"Error in ensemble prediction: {e}")

        # Use ensemble score if available, otherwise average of individual models
        final_confidence = (
            ensemble_confidence
            if ensemble_confidence > 0
            else np.mean(list(predictions.values()))
        )

        # Authentication decision
        threshold = self.auth_thresholds.get(user_id, self.default_threshold)
        authenticated = final_confidence >= threshold

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "authenticated": authenticated,
            "confidence": final_confidence,
            "threshold_used": threshold,
            "model_predictions": confidence_scores,
            "processing_time_ms": processing_time,
            "reason": "Authentication successful"
            if authenticated
            else "Confidence below threshold",
        }

    def _calculate_optimal_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate optimal authentication threshold using EER (Equal Error Rate)"""

        if self.ensemble_model is None:
            return self.default_threshold

        # Get prediction probabilities
        proba_array = self.ensemble_model.predict_proba(X)

        # Handle both binary and single-class scenarios
        if proba_array.shape[1] > 1:
            # Binary classification - get positive class probabilities
            probabilities = proba_array[:, 1]
        else:
            # Single class - use the only available probabilities
            probabilities = proba_array[:, 0]

        # Calculate EER
        thresholds = np.linspace(0.1, 0.9, 100)
        far_rates = []  # False Acceptance Rate
        frr_rates = []  # False Rejection Rate

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)

            # Calculate error rates
            tn = np.sum((y == 0) & (predictions == 0))
            fp = np.sum((y == 0) & (predictions == 1))
            fn = np.sum((y == 1) & (predictions == 0))
            tp = np.sum((y == 1) & (predictions == 1))

            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0

            far_rates.append(far)
            frr_rates.append(frr)

        # Find EER point
        eer_errors = [abs(far - frr) for far, frr in zip(far_rates, frr_rates)]
        eer_index = np.argmin(eer_errors)

        optimal_threshold = thresholds[eer_index]

        logger.info(
            f"Optimal threshold: {optimal_threshold:.3f} (EER: {eer_errors[eer_index]:.3f})"
        )

        return optimal_threshold

    def _save_user_model(self, user_id: str, models: Dict, scaler: StandardScaler):
        """Save trained models and preprocessing objects"""

        user_model_path = self.model_save_path / f"{user_id}_models.joblib"

        model_data = {
            "models": models,
            "scaler": scaler,
            "trained_at": datetime.utcnow().isoformat(),
            "user_id": user_id,
        }

        joblib.dump(model_data, user_model_path)
        logger.info(f"Saved model for user {user_id} to {user_model_path}")

    def train_device_type_model(
        self, feature_vectors: List[List[float]], labels: List[str]
    ) -> Dict[str, Any]:
        """Train a global classifier to distinguish numpad vs top-row input"""

        if len(feature_vectors) < 4:
            raise ValueError(
                "Insufficient data for device type training. Need at least 4 samples."
            )

        unique_labels = set(labels)
        if len(unique_labels) < 2:
            raise ValueError(
                "Device type classifier requires samples from at least two input types"
            )

        logger.info(
            "Training device type classifier with %d samples (%s)",
            len(feature_vectors),
            unique_labels,
        )

        X = np.array(feature_vectors)
        y_encoded = self.device_type_label_encoder.fit_transform(labels)
        X_scaled = self.device_type_scaler.fit_transform(X)

        class_counts = np.bincount(y_encoded)
        min_class_samples = class_counts.min() if len(class_counts) > 0 else 0
        use_cross_validation = min_class_samples >= 2 and len(X_scaled) >= 4
        n_folds = 2

        if use_cross_validation:
            n_folds = min(5, len(X_scaled), int(min_class_samples))
            if n_folds < 2:
                use_cross_validation = False

        cv = (
            StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            if use_cross_validation
            else None
        )

        model_scores: Dict[str, Dict[str, Any]] = {}
        trained_models: Dict[str, Any] = {}

        for model_name, model in self.models.items():
            try:
                model_clone = model.__class__(**model.get_params())

                if model_name == "knn":
                    n_neighbors = model_clone.get_params()["n_neighbors"]
                    max_neighbors = len(X_scaled) - 1
                    if max_neighbors >= 1 and n_neighbors > max_neighbors:
                        model_clone.set_params(n_neighbors=max(1, max_neighbors))

                model_clone.fit(X_scaled, y_encoded)

                if use_cross_validation and cv is not None:
                    cv_scores = cross_val_score(
                        model_clone,
                        X_scaled,
                        y_encoded,
                        cv=cv,
                        scoring="accuracy",
                    )
                    mean_score = float(cv_scores.mean())
                    std_score = float(cv_scores.std())
                else:
                    mean_score = float(model_clone.score(X_scaled, y_encoded))
                    std_score = 0.0

                model_scores[model_name] = {
                    "cv_mean": mean_score,
                    "cv_std": std_score,
                    "model": model_clone,
                }
                trained_models[model_name] = model_clone

                logger.info(
                    "Device type model %s accuracy: %.3f (cv=%s)",
                    model_name,
                    mean_score,
                    "yes" if use_cross_validation else "no",
                )

            except Exception as exc:
                logger.error("Error training device type model %s: %s", model_name, exc)

        # Build ensemble if possible
        ensemble_estimators = [
            (name, info["model"]) for name, info in model_scores.items()
        ]

        if ensemble_estimators:
            ensemble = VotingClassifier(estimators=ensemble_estimators, voting="soft")
            ensemble.fit(X_scaled, y_encoded)

            if use_cross_validation and cv is not None:
                ensemble_scores = cross_val_score(
                    ensemble, X_scaled, y_encoded, cv=cv, scoring="accuracy"
                )
                ensemble_mean = float(ensemble_scores.mean())
                ensemble_std = float(ensemble_scores.std())
            else:
                ensemble_mean = float(ensemble.score(X_scaled, y_encoded))
                ensemble_std = 0.0

            model_scores["ensemble"] = {
                "cv_mean": ensemble_mean,
                "cv_std": ensemble_std,
                "model": ensemble,
            }
            trained_models["ensemble"] = ensemble
            self.device_type_ensemble = ensemble

        class_distribution = {
            self.device_type_label_encoder.inverse_transform([idx])[0]: int(count)
            for idx, count in enumerate(class_counts)
        }

        self.device_type_models = trained_models
        self.device_type_model_performance = {
            name: info["cv_mean"] for name, info in model_scores.items()
        }

        self._save_device_type_model(
            trained_models, self.device_type_scaler, self.device_type_label_encoder
        )

        return {
            "samples_trained": len(X_scaled),
            "class_distribution": class_distribution,
            "model_scores": {
                name: info["cv_mean"] for name, info in model_scores.items()
            },
            "cross_validated": use_cross_validation,
        }

    def _save_device_type_model(
        self,
        models: Dict[str, Any],
        scaler: StandardScaler,
        label_encoder: LabelEncoder,
    ):
        """Persist the global device type classifier"""

        model_data = {
            "models": models,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "trained_at": datetime.utcnow().isoformat(),
            "performance": self.device_type_model_performance,
        }

        joblib.dump(model_data, self.device_type_model_path)
        logger.info("Saved device type model to %s", self.device_type_model_path)

    def _load_device_type_model(self) -> Optional[Dict[str, Any]]:
        """Load the device type classifier from disk if not cached"""

        if self.device_type_models:
            return {
                "models": self.device_type_models,
                "scaler": self.device_type_scaler,
                "label_encoder": self.device_type_label_encoder,
            }

        if not self.device_type_model_path.exists():
            return None

        try:
            model_data = joblib.load(self.device_type_model_path)
            self.device_type_models = model_data.get("models", {})
            self.device_type_scaler = model_data.get("scaler", StandardScaler())
            self.device_type_label_encoder = model_data.get(
                "label_encoder", LabelEncoder()
            )
            self.device_type_model_performance = model_data.get("performance", {})
            self.device_type_ensemble = self.device_type_models.get("ensemble")

            return {
                "models": self.device_type_models,
                "scaler": self.device_type_scaler,
                "label_encoder": self.device_type_label_encoder,
            }

        except Exception as exc:
            logger.error("Failed to load device type model: %s", exc)
            self.device_type_models = {}
            self.device_type_ensemble = None
            return None

    def classify_device_type(
        self, feature_vector: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Predict whether a pattern was typed on the numpad or key row"""

        model_data = self._load_device_type_model()
        if not model_data:
            return None

        models = model_data.get("models", {})
        if not models:
            return None

        scaler: StandardScaler = model_data["scaler"]
        label_encoder: LabelEncoder = model_data["label_encoder"]

        X_scaled = scaler.transform([feature_vector])

        probabilities: Dict[str, float] = {}
        confidence = 0.0
        predicted_label: Optional[str] = None
        model_used = "ensemble"

        if "ensemble" in models:
            ensemble_model = models["ensemble"]
            proba = ensemble_model.predict_proba(X_scaled)[0]
            for idx, value in enumerate(proba):
                label = label_encoder.inverse_transform([idx])[0]
                probabilities[label] = float(value)
            best_index = int(np.argmax(proba))
            predicted_label = label_encoder.inverse_transform([best_index])[0]
            confidence = float(proba[best_index])
        else:
            aggregate = np.zeros(len(label_encoder.classes_), dtype=float)
            valid_models = 0

            for model_name, model in models.items():
                try:
                    proba = model.predict_proba(X_scaled)[0]
                    aggregate += proba
                    valid_models += 1
                except Exception as exc:
                    logger.error(
                        "Device type model %s failed to produce probabilities: %s",
                        model_name,
                        exc,
                    )

            if valid_models == 0:
                return None

            aggregate /= valid_models
            for idx, value in enumerate(aggregate):
                label = label_encoder.inverse_transform([idx])[0]
                probabilities[label] = float(value)
            best_index = int(np.argmax(aggregate))
            predicted_label = label_encoder.inverse_transform([best_index])[0]
            confidence = float(aggregate[best_index])
            model_used = "average"

        return {
            "predicted_device_type": predicted_label,
            "confidence": confidence,
            "model_used": model_used,
            "probabilities": probabilities,
        }

    def _load_user_model(self, user_id: str) -> Optional[Dict]:
        """Load trained models for a user"""

        user_model_path = self.model_save_path / f"{user_id}_models.joblib"

        if not user_model_path.exists():
            logger.warning(f"No model found for user {user_id}")
            return None

        try:
            model_data = joblib.load(user_model_path)
            logger.info(f"Loaded model for user {user_id}")
            return model_data
        except Exception as e:
            logger.error(f"Error loading model for user {user_id}: {e}")
            return None

    def get_model_performance(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for user's model"""
        perf = self.model_performance.get(user_id)
        if not perf:
            return None

        # Extract only JSON-serializable data (exclude model objects)
        json_safe_perf = {}
        for model_name, info in perf.items():
            json_safe_perf[model_name] = {
                "cv_mean": float(info["cv_mean"])
                if not np.isnan(info["cv_mean"])
                else 0.0,
                "cv_std": float(info["cv_std"])
                if not np.isnan(info["cv_std"])
                else 0.0,
            }

        return json_safe_perf

    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter tuning for better model performance"""

        logger.info("Starting hyperparameter tuning...")

        # Calculate max neighbors for KNN based on available samples
        n_samples = len(X)
        max_neighbors = min(
            11, max(1, n_samples - 1)
        )  # Ensure at least 1, max n_samples-1

        # Dynamically create valid neighbor values
        valid_neighbors = [n for n in [3, 5, 7, 9, 11] if n <= max_neighbors]
        if not valid_neighbors:
            valid_neighbors = [min(3, max_neighbors)]  # Fallback to minimum

        # Parameter grids for each model
        param_grids = {
            "svm": {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                "kernel": ["rbf", "poly"],
            },
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "knn": {
                "n_neighbors": valid_neighbors,
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 6, 9],
            },
        }

        logger.info(
            f"Using n_neighbors values for KNN: {valid_neighbors} (max allowed: {max_neighbors})"
        )

        tuned_models = {}
        cv = StratifiedKFold(
            n_splits=min(5, n_samples // 2), shuffle=True, random_state=42
        )  # Adjust CV folds

        for model_name, base_model in self.models.items():
            if model_name in param_grids:
                logger.info(f"Tuning {model_name}...")

                grid_search = GridSearchCV(
                    base_model,
                    param_grids[model_name],
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=0,
                )

                grid_search.fit(X, y)

                tuned_models[model_name] = {
                    "best_model": grid_search.best_estimator_,
                    "best_params": grid_search.best_params_,
                    "best_score": grid_search.best_score_,
                }

                logger.info(f"{model_name} best score: {grid_search.best_score_:.3f}")
                logger.info(f"{model_name} best params: {grid_search.best_params_}")

        # Update models with tuned versions
        for model_name, model_info in tuned_models.items():
            self.models[model_name] = model_info["best_model"]

        return tuned_models
