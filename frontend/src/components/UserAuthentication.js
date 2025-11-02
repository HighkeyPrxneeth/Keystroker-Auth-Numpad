import React, { useState, useEffect } from "react";
import { useKeystrokeCapture } from "../hooks/useKeystrokeCapture";
import apiService from "../services/api";
import {
  INPUT_DEVICE_TYPES,
  getRandomPhrase,
  getDeviceTypeLabel,
} from "../constants/typingPhrases";

const UserAuthentication = () => {
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [inputDeviceType, setInputDeviceType] = useState(
    INPUT_DEVICE_TYPES.KEYROW
  );
  const [testPhrase, setTestPhrase] = useState(
    getRandomPhrase(INPUT_DEVICE_TYPES.KEYROW)
  );
  const [typedText, setTypedText] = useState("");
  const [status, setStatus] = useState({ type: "", message: "" });
  const [loading, setLoading] = useState(false);
  const [authResult, setAuthResult] = useState(null);
  const [deviceClassification, setDeviceClassification] = useState(null);
  const [lastAttemptDeviceType, setLastAttemptDeviceType] = useState(null);

  const {
    keystrokeData,
    isCapturing,
    startCapture,
    stopCapture,
    handleKeyDown,
    handleKeyUp,
    getTypingPattern,
  } = useKeystrokeCapture();

  useEffect(() => {
    loadUsers();
  }, []);

  const loadUsers = async () => {
    try {
      const userList = await apiService.getAllUsers();
      setUsers(userList);
    } catch (error) {
      console.error("Failed to load users:", error);
    }
  };

  const handleStartAuth = () => {
    setTypedText("");
    setStatus({ type: "", message: "" });
    setAuthResult(null);
    setDeviceClassification(null);
    setLastAttemptDeviceType(null);
    setTestPhrase(getRandomPhrase(inputDeviceType));
    startCapture();
    setStatus({
      type: "info",
      message: `Recording keystroke pattern using ${getDeviceTypeLabel(
        inputDeviceType
      )}. Type the phrase exactly as shown.`,
    });
  };

  const handleDeviceTypeChange = (event) => {
    const nextType = event.target.value;
    setInputDeviceType(nextType);
    setTypedText("");
    setTestPhrase(getRandomPhrase(nextType));
    stopCapture();
    setStatus({
      type: "info",
      message: `Device target updated. Click 'Start Authentication' and use the ${getDeviceTypeLabel(
        nextType
      ).toLowerCase()}.`,
    });
  };

  const handleTextChange = (e) => {
    const value = e.target.value;
    setTypedText(value);

    // Check if user completed the phrase
    if (value === testPhrase && isCapturing) {
      stopCapture();
      setStatus({
        type: "info",
        message: 'Pattern captured! Click "Authenticate" to verify identity.',
      });
    }
  };

  const handleAuthenticate = async () => {
    if (!selectedUserId) {
      setStatus({
        type: "error",
        message: "Please select a user to authenticate.",
      });
      return;
    }

    if (keystrokeData.length === 0) {
      setStatus({
        type: "error",
        message: "No keystroke data captured. Please type the phrase first.",
      });
      return;
    }

    setLoading(true);
    try {
      // Find the selected user to get their username
      const selectedUser = users.find((user) => user.id === selectedUserId);
      if (!selectedUser) {
        setStatus({
          type: "error",
          message: "Selected user not found.",
        });
        setLoading(false);
        return;
      }

      const pattern = getTypingPattern();

      // Convert keystroke data to the format expected by backend
      // The backend expects keydown and keyup events, but our hook only captures keyup
      // So we'll simulate both events with proper timing
      const events = [];
      pattern.keystrokes.forEach((keystroke) => {
        // Add keydown event (before the keyup)
        events.push({
          key: keystroke.key,
          key_code: keystroke.keyCode || 0,
          timestamp: keystroke.timestamp - keystroke.dwellTime, // Start of key press
          event_type: "keydown",
        });

        // Add keyup event
        events.push({
          key: keystroke.key,
          key_code: keystroke.keyCode || 0,
          timestamp: keystroke.timestamp, // End of key press
          event_type: "keyup",
        });
      });

      const attemptDeviceType = inputDeviceType;
      const authData = {
        username: selectedUser.username,
        pattern: {
          text_typed: typedText,
          events: events,
          input_device_type: attemptDeviceType,
        },
      };

      const result = await apiService.authenticateUser(authData);
      setAuthResult(result);
      setDeviceClassification({
        predicted: result.predicted_device_type || null,
        confidence:
          typeof result.device_type_confidence === "number"
            ? result.device_type_confidence
            : null,
      });
      setLastAttemptDeviceType(attemptDeviceType);

      // The backend returns status: "success" or "failed" and confidence_score
      const isAuthenticated = result.status === "success";
      const confidence = result.confidence_score || 0;

      if (isAuthenticated) {
        setStatus({
          type: "success",
          message: `Authentication successful! Confidence: ${(
            confidence * 100
          ).toFixed(1)}%`,
        });
      } else {
        setStatus({
          type: "error",
          message: `Authentication failed. Confidence: ${(
            confidence * 100
          ).toFixed(1)}%`,
        });
      }

      setTypedText("");
    } catch (error) {
      let errorMessage = error.message || "Authentication failed";

      // Check for specific enrollment error
      if (
        errorMessage.includes("User not enrolled") ||
        errorMessage.includes("not enrolled")
      ) {
        errorMessage =
          "User has not completed enrollment yet. Please complete the enrollment process first.";
      }

      setStatus({
        type: "error",
        message: errorMessage,
      });
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return "#28a745";
    if (confidence >= 0.6) return "#ffc107";
    return "#dc3545";
  };

  return (
    <div className="card">
      <h2>🔐 User Authentication</h2>
      <p>
        Type the phrase to authenticate using your unique keystroke pattern.
      </p>

      {status.message && (
        <div className={`status-message status-${status.type}`}>
          {status.message}
        </div>
      )}

      <div className="form-group">
        <label htmlFor="userSelect">Select User to Authenticate</label>
        <select
          id="userSelect"
          value={selectedUserId}
          onChange={(e) => setSelectedUserId(e.target.value)}
          style={{
            width: "100%",
            padding: "0.75rem",
            borderRadius: "8px",
            border: "2px solid #e1e5e9",
          }}
        >
          <option value="">Choose a user...</option>
          {users.map((user) => (
            <option key={user.id} value={user.id}>
              {user.username} ({user.email})
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Input Device Target</label>
        <div style={{ display: "flex", gap: "1rem" }}>
          <label
            style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
          >
            <input
              type="radio"
              name="authDeviceType"
              value={INPUT_DEVICE_TYPES.KEYROW}
              checked={inputDeviceType === INPUT_DEVICE_TYPES.KEYROW}
              onChange={handleDeviceTypeChange}
            />
            Top-row number keys
          </label>
          <label
            style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
          >
            <input
              type="radio"
              name="authDeviceType"
              value={INPUT_DEVICE_TYPES.NUMPAD}
              checked={inputDeviceType === INPUT_DEVICE_TYPES.NUMPAD}
              onChange={handleDeviceTypeChange}
            />
            Numeric keypad
          </label>
        </div>
        <small style={{ color: "#4a5568" }}>
          Select the device you will use so the classifier can learn from both
          input sources.
        </small>
      </div>

      <div className="form-group">
        <label htmlFor="testPhrase">Authentication Phrase</label>
        <input
          type="text"
          id="testPhrase"
          value={testPhrase}
          readOnly
          placeholder="Authentication phrase will randomize for each attempt"
        />
        <button
          type="button"
          className="btn btn-secondary"
          style={{ marginTop: "0.5rem" }}
          onClick={() => {
            stopCapture();
            setTypedText("");
            setTestPhrase(getRandomPhrase(inputDeviceType, testPhrase));
            setStatus({
              type: "info",
              message:
                "Phrase updated. Restart recording to capture a new attempt.",
            });
          }}
        >
          Shuffle Phrase
        </button>
      </div>

      <div className="form-group">
        <label htmlFor="typingArea">Type Here</label>
        <textarea
          id="typingArea"
          className="typing-area"
          value={typedText}
          onChange={handleTextChange}
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          placeholder="Click 'Start Authentication' then type the phrase above..."
          disabled={!isCapturing && keystrokeData.length === 0}
        />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <p>
          <strong>Target:</strong> {testPhrase}
        </p>
        <p>
          <strong>Typed:</strong> {typedText}
        </p>
        <p>
          <strong>Progress:</strong> {typedText.length}/{testPhrase.length}{" "}
          characters
        </p>
        {typedText === testPhrase && typedText.length > 0 && (
          <p style={{ color: "green", fontWeight: "bold" }}>
            ✓ Phrase completed!
          </p>
        )}
      </div>

      <div>
        <button
          className="btn"
          onClick={handleStartAuth}
          disabled={isCapturing || !selectedUserId}
        >
          Start Authentication
        </button>

        <button
          className="btn btn-secondary"
          onClick={stopCapture}
          disabled={!isCapturing}
        >
          Stop Recording
        </button>

        <button
          className="btn btn-success"
          onClick={handleAuthenticate}
          disabled={loading || keystrokeData.length === 0 || !selectedUserId}
        >
          {loading ? "Authenticating..." : "Authenticate"}
        </button>
      </div>

      {authResult && (
        <div
          className="card"
          style={{
            marginTop: "2rem",
            backgroundColor:
              authResult.status === "success" ? "#d4edda" : "#f8d7da",
          }}
        >
          <h3>Authentication Result</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div
                className="stat-value"
                style={{
                  color:
                    authResult.status === "success" ? "#28a745" : "#dc3545",
                }}
              >
                {authResult.status === "success" ? "✓" : "✗"}
              </div>
              <div className="stat-label">Status</div>
            </div>
            <div className="stat-card">
              <div
                className="stat-value"
                style={{
                  color: getConfidenceColor(authResult.confidence_score || 0),
                }}
              >
                {((authResult.confidence_score || 0) * 100).toFixed(1)}%
              </div>
              <div className="stat-label">Confidence</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {authResult.processing_time_ms?.toFixed(1) || 0} ms
              </div>
              <div className="stat-label">Processing Time</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {authResult.message || "No message"}
              </div>
              <div className="stat-label">Details</div>
            </div>
            {deviceClassification?.predicted && (
              <div className="stat-card">
                <div className="stat-value">
                  {getDeviceTypeLabel(deviceClassification.predicted)}
                </div>
                <div className="stat-label">
                  Device Prediction
                  {typeof deviceClassification.confidence === "number" && (
                    <span
                      style={{
                        display: "block",
                        fontSize: "0.8rem",
                        marginTop: "0.25rem",
                      }}
                    >
                      Confidence{" "}
                      {(deviceClassification.confidence * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
          {deviceClassification?.predicted && (
            <p style={{ marginTop: "1rem", fontSize: "0.9rem" }}>
              You indicated{" "}
              <strong>
                {getDeviceTypeLabel(lastAttemptDeviceType || inputDeviceType)}
              </strong>
              ; model detected{" "}
              <strong>
                {getDeviceTypeLabel(deviceClassification.predicted)}
              </strong>
              .
            </p>
          )}
        </div>
      )}

      {keystrokeData.length > 0 && (
        <div className="keystroke-display">
          <h4>Captured Keystrokes: {keystrokeData.length}</h4>
          <div style={{ maxHeight: "150px", overflowY: "auto" }}>
            {keystrokeData.map((keystroke, index) => (
              <div key={index}>
                Key: {keystroke.key} | Dwell: {keystroke.dwellTime}ms | Time:{" "}
                {keystroke.timestamp}ms
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default UserAuthentication;
