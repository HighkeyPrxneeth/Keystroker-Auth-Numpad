import React, { useState, useEffect, useCallback } from "react";
import { useKeystrokeCapture } from "../hooks/useKeystrokeCapture";
import apiService from "../services/api";
import {
  INPUT_DEVICE_TYPES,
  getRandomPhrase,
  getDeviceTypeLabel,
} from "../constants/typingPhrases";

const UserEnrollment = ({ currentUser, setCurrentUser }) => {
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
  const [enrollmentPatterns, setEnrollmentPatterns] = useState([]);
  const [currentPatternNumber, setCurrentPatternNumber] = useState(1);
  const [deviceTypeCounts, setDeviceTypeCounts] = useState({
    [INPUT_DEVICE_TYPES.KEYROW]: 0,
    [INPUT_DEVICE_TYPES.NUMPAD]: 0,
  });
  const requiredPatterns = 5;

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

  useEffect(() => {
    if (currentUser) {
      setSelectedUserId(currentUser.id);
    }
  }, [currentUser]);

  const prepareForCapture = useCallback(
    (deviceType, patternNumber = currentPatternNumber) => {
      const nextPhrase = getRandomPhrase(deviceType, testPhrase);
      setTestPhrase(nextPhrase);
      setTypedText("");
      startCapture();
      setStatus({
        type: "info",
        message: `Recording pattern ${patternNumber}/${requiredPatterns} using ${getDeviceTypeLabel(
          deviceType
        )}. Type the phrase exactly as shown.`,
      });
    },
    [currentPatternNumber, requiredPatterns, startCapture, testPhrase]
  );

  const handleDeviceTypeChange = (event) => {
    const nextType = event.target.value;
    setInputDeviceType(nextType);
    setTypedText("");
    setTestPhrase(getRandomPhrase(nextType));
    stopCapture();
    setStatus({
      type: "info",
      message: `Device target updated. Please click '${
        enrollmentPatterns.length === 0
          ? "Start Enrollment"
          : "Start Next Pattern"
      }' and type the ${getDeviceTypeLabel(nextType).toLowerCase()}.`,
    });
  };

  const loadUsers = async () => {
    try {
      const userList = await apiService.getAllUsers();
      setUsers(userList);
    } catch (error) {
      console.error("Failed to load users:", error);
    }
  };

  const handleStartEnrollment = () => {
    setTypedText("");
    setStatus({ type: "", message: "" });
    setEnrollmentPatterns([]);
    setCurrentPatternNumber(1);
    setDeviceTypeCounts({
      [INPUT_DEVICE_TYPES.KEYROW]: 0,
      [INPUT_DEVICE_TYPES.NUMPAD]: 0,
    });
    prepareForCapture(inputDeviceType, 1);
  };

  const handleTextChange = (e) => {
    const value = e.target.value;
    setTypedText(value);

    // Check if user completed the phrase
    if (value === testPhrase && isCapturing) {
      stopCapture();
      setStatus({
        type: "info",
        message: 'Pattern captured! Click "Submit Pattern" to save.',
      });
    }
  };

  const handleSubmitPattern = async () => {
    if (!selectedUserId) {
      setStatus({
        type: "error",
        message: "Please select a user first.",
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

    try {
      const pattern = getTypingPattern();

      // Convert keystroke data to the format expected by backend
      const events = [];
      pattern.keystrokes.forEach((keystroke) => {
        // Add keydown event
        events.push({
          key: keystroke.key,
          key_code: keystroke.keyCode || 0,
          timestamp: keystroke.timestamp - keystroke.dwellTime,
          event_type: "keydown",
        });

        // Add keyup event
        events.push({
          key: keystroke.key,
          key_code: keystroke.keyCode || 0,
          timestamp: keystroke.timestamp,
          event_type: "keyup",
        });
      });

      // Store the pattern locally
      const patternData = {
        text_typed: typedText,
        events: events,
        input_device_type: inputDeviceType,
      };

      const updatedPatterns = [...enrollmentPatterns, patternData];
      setEnrollmentPatterns(updatedPatterns);
      const newPatternCount = updatedPatterns.length;
      setDeviceTypeCounts((prev) => ({
        ...prev,
        [inputDeviceType]: prev[inputDeviceType] + 1,
      }));
      setCurrentPatternNumber(newPatternCount + 1);

      // Reset for next pattern
      setTypedText("");

      if (newPatternCount >= requiredPatterns) {
        setStatus({
          type: "success",
          message: `Pattern ${newPatternCount}/${requiredPatterns} captured! You can now complete enrollment.`,
        });
      } else {
        setStatus({
          type: "info",
          message: `Pattern ${newPatternCount}/${requiredPatterns} captured for ${getDeviceTypeLabel(
            inputDeviceType
          ).toLowerCase()}. Choose the next device type and click "Start Next Pattern" to continue.`,
        });
      }
    } catch (error) {
      setStatus({
        type: "error",
        message: error.message || "Failed to capture pattern",
      });
    }
  };

  const handleStartNextPattern = () => {
    setTypedText("");
    prepareForCapture(inputDeviceType, currentPatternNumber);
  };

  const handleCompleteEnrollment = async () => {
    if (enrollmentPatterns.length < requiredPatterns) {
      setStatus({
        type: "error",
        message: `Need ${requiredPatterns} patterns for enrollment. You have ${enrollmentPatterns.length}.`,
      });
      return;
    }

    if (
      deviceTypeCounts[INPUT_DEVICE_TYPES.KEYROW] === 0 ||
      deviceTypeCounts[INPUT_DEVICE_TYPES.NUMPAD] === 0
    ) {
      setStatus({
        type: "error",
        message:
          "Please record at least one pattern using both the top-row numbers and the numeric keypad to train the device classifier.",
      });
      return;
    }

    const selectedUser = users.find((user) => user.id === selectedUserId);
    if (!selectedUser) {
      setStatus({
        type: "error",
        message: "Selected user not found.",
      });
      return;
    }

    setLoading(true);
    try {
      const enrollmentData = {
        username: selectedUser.username,
        email: selectedUser.email,
        patterns: enrollmentPatterns,
      };

      const result = await apiService.enrollUser(enrollmentData);

      setStatus({
        type: "success",
        message: `Enrollment successful! Processed ${result.patterns_processed} patterns.`,
      });

      // Update current user if it's the enrolled user
      if (currentUser && currentUser.id === selectedUserId) {
        setCurrentUser({
          ...currentUser,
          is_enrolled: true,
          enrollment_patterns_count: result.patterns_processed,
        });
      }

      // Reset everything
      setEnrollmentPatterns([]);
      setCurrentPatternNumber(1);
      setTypedText("");
    } catch (error) {
      setStatus({
        type: "error",
        message: error.message || "Enrollment failed",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>� User Enrollment</h2>
      <p>
        Collect multiple keystroke patterns to train the authentication model.
      </p>

      {status.message && (
        <div className={`status-message status-${status.type}`}>
          {status.message}
        </div>
      )}

      <div className="form-group">
        <label htmlFor="userSelect">Select User to Enroll</label>
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
              name="inputDeviceType"
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
              name="inputDeviceType"
              value={INPUT_DEVICE_TYPES.NUMPAD}
              checked={inputDeviceType === INPUT_DEVICE_TYPES.NUMPAD}
              onChange={handleDeviceTypeChange}
            />
            Numeric keypad
          </label>
        </div>
        <small style={{ color: "#4a5568" }}>
          Capture patterns from both device types to help the classifier
          distinguish input sources.
        </small>
      </div>

      <div className="form-group">
        <label htmlFor="testPhrase">Enrollment Phrase</label>
        <input
          type="text"
          id="testPhrase"
          value={testPhrase}
          readOnly
          placeholder="Enrollment phrase will randomize for each capture"
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
                "Phrase updated. Restart recording to capture a new pattern.",
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
          placeholder="Click 'Start Enrollment' or 'Start Next Pattern' then type the phrase above..."
          disabled={!isCapturing}
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
        <p>
          <strong>Patterns Collected:</strong> {enrollmentPatterns.length}/
          {requiredPatterns}
        </p>
        <p>
          <strong>Device Coverage:</strong> Top row{" "}
          {deviceTypeCounts[INPUT_DEVICE_TYPES.KEYROW]} / Numpad{" "}
          {deviceTypeCounts[INPUT_DEVICE_TYPES.NUMPAD]}
        </p>
        {typedText === testPhrase && typedText.length > 0 && (
          <p style={{ color: "green", fontWeight: "bold" }}>
            ✓ Phrase completed!
          </p>
        )}
      </div>

      <div>
        {enrollmentPatterns.length === 0 ? (
          <button
            className="btn"
            onClick={handleStartEnrollment}
            disabled={!selectedUserId}
          >
            Start Enrollment
          </button>
        ) : (
          <button
            className="btn"
            onClick={handleStartNextPattern}
            disabled={
              isCapturing || enrollmentPatterns.length >= requiredPatterns
            }
          >
            Start Next Pattern
          </button>
        )}

        <button
          className="btn btn-secondary"
          onClick={stopCapture}
          disabled={!isCapturing}
        >
          Stop Recording
        </button>

        <button
          className="btn btn-success"
          onClick={handleSubmitPattern}
          disabled={keystrokeData.length === 0 || isCapturing}
        >
          Submit Pattern
        </button>

        <button
          className="btn btn-primary"
          onClick={handleCompleteEnrollment}
          disabled={loading || enrollmentPatterns.length < requiredPatterns}
        >
          {loading ? "Enrolling..." : "Complete Enrollment"}
        </button>
      </div>

      {enrollmentPatterns.length > 0 && (
        <div
          className="card"
          style={{ marginTop: "2rem", backgroundColor: "#f8f9fa" }}
        >
          <h3>Enrollment Progress</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{enrollmentPatterns.length}</div>
              <div className="stat-label">Patterns Collected</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {requiredPatterns - enrollmentPatterns.length}
              </div>
              <div className="stat-label">Patterns Remaining</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {enrollmentPatterns.length >= requiredPatterns ? "✓" : "○"}
              </div>
              <div className="stat-label">Ready to Enroll</div>
            </div>
          </div>
        </div>
      )}

      {keystrokeData.length > 0 && (
        <div className="keystroke-display">
          <h4>Current Pattern Keystrokes: {keystrokeData.length}</h4>
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

export default UserEnrollment;
