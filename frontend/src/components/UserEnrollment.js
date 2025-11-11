import React, { useState, useEffect, useCallback } from "react";
import { useKeystrokeCapture } from "../hooks/useKeystrokeCapture";
import apiService from "../services/api";
import {
  INPUT_DEVICE_TYPES,
  getRandomPhrase,
  getDeviceTypeLabel,
  getDeviceGuidance,
} from "../constants/typingPhrases";

const PATTERNS_PER_DEVICE = 3;
const REQUIRED_PATTERNS = PATTERNS_PER_DEVICE * 2;

const UserEnrollment = ({ currentUser, setCurrentUser }) => {
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [inputDeviceType, setInputDeviceType] = useState(
    INPUT_DEVICE_TYPES.KEYROW
  );
  const [testPhrase, setTestPhrase] = useState(getRandomPhrase());
  const [typedText, setTypedText] = useState("");
  const [status, setStatus] = useState({ type: "", message: "" });
  const [loading, setLoading] = useState(false);
  const [enrollmentPatterns, setEnrollmentPatterns] = useState([]);
  const [currentPatternNumber, setCurrentPatternNumber] = useState(1);
  const [deviceTypeCounts, setDeviceTypeCounts] = useState({
    [INPUT_DEVICE_TYPES.KEYROW]: 0,
    [INPUT_DEVICE_TYPES.NUMPAD]: 0,
  });

  const {
    keystrokeData,
    isCapturing,
    startCapture,
    stopCapture,
    handleKeyDown,
    handleKeyUp,
    getTypingPattern,
  } = useKeystrokeCapture();

  const loadUsers = useCallback(async () => {
    try {
      const userList = await apiService.getAllUsers();
      setUsers(userList);
    } catch (error) {
      console.error("Failed to load users:", error);
    }
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  useEffect(() => {
    if (currentUser) {
      setSelectedUserId(currentUser.id);
      loadUsers();
    }
  }, [currentUser, loadUsers]);

  const getTargetDeviceForPattern = useCallback((patternNumber) => {
    return patternNumber <= PATTERNS_PER_DEVICE
      ? INPUT_DEVICE_TYPES.KEYROW
      : INPUT_DEVICE_TYPES.NUMPAD;
  }, []);

  const prepareForCapture = useCallback(
    (patternNumber = currentPatternNumber) => {
      const deviceType = getTargetDeviceForPattern(patternNumber);
      const guidance = getDeviceGuidance(deviceType);
      setInputDeviceType(deviceType);
      setTestPhrase((prev) => getRandomPhrase(prev.text || prev));
      setTypedText("");
      startCapture();
      setStatus({
        type: "info",
        message: `Recording pattern ${patternNumber}/${REQUIRED_PATTERNS} using ${getDeviceTypeLabel(
          deviceType
        )}. ${guidance} Each digit must come from the correct section of the keyboard.`,
      });
    },
    [currentPatternNumber, getTargetDeviceForPattern, startCapture]
  );

  const handleStartEnrollment = () => {
    stopCapture();
    setTypedText("");
    setStatus({ type: "", message: "" });
    setEnrollmentPatterns([]);
    setCurrentPatternNumber(1);
    setDeviceTypeCounts({
      [INPUT_DEVICE_TYPES.KEYROW]: 0,
      [INPUT_DEVICE_TYPES.NUMPAD]: 0,
    });
    prepareForCapture(1);
  };

  const handleTextChange = (e) => {
    const value = e.target.value;
    setTypedText(value);

    // Check if user completed the phrase
    const targetText =
      typeof testPhrase === "string" ? testPhrase : testPhrase.text;
    if (value === targetText && isCapturing) {
      stopCapture();
      setStatus({
        type: "info",
        message: 'Pattern captured! Click "Submit Pattern" to save.',
      });
    }
  };

  const handleKeyDownWithValidation = useCallback(
    (event) => {
      if (!isCapturing) {
        return;
      }

      const isDigitKey = event.key.length === 1 && /[0-9]/.test(event.key);
      if (isDigitKey) {
        const isNumpadEvent = event.location === 3;
        const usingTopRow = inputDeviceType === INPUT_DEVICE_TYPES.KEYROW;
        const usingNumpad = inputDeviceType === INPUT_DEVICE_TYPES.NUMPAD;

        if ((usingTopRow && isNumpadEvent) || (usingNumpad && !isNumpadEvent)) {
          event.preventDefault();
          event.stopPropagation();
          setStatus({
            type: "error",
            message: usingTopRow
              ? "❌ Use the digits above the letters (0-9 keys in the top row); the numpad is disabled for this pattern."
              : "❌ Use the numeric keypad on the right side; the top-row digits are disabled for this pattern.",
          });
          return;
        }
      }

      handleKeyDown(event);
    },
    [handleKeyDown, inputDeviceType, isCapturing]
  );

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

      // Store the pattern locally with device type
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
      const nextPatternNumber = newPatternCount + 1;
      setCurrentPatternNumber(nextPatternNumber);

      // Reset for next pattern
      setTypedText("");

      if (newPatternCount >= REQUIRED_PATTERNS) {
        setStatus({
          type: "success",
          message: `Pattern ${newPatternCount}/${REQUIRED_PATTERNS} captured! You can now complete enrollment.`,
        });
      } else {
        const nextDeviceType = getTargetDeviceForPattern(nextPatternNumber);
        setInputDeviceType(nextDeviceType);
        setStatus({
          type: "info",
          message: `Pattern ${newPatternCount}/${REQUIRED_PATTERNS} captured. Next: ${getDeviceTypeLabel(
            nextDeviceType
          )}. ${getDeviceGuidance(
            nextDeviceType
          )} Click "Start Next Pattern" to continue.`,
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
    stopCapture();
    prepareForCapture(currentPatternNumber);
  };

  const handleCompleteEnrollment = async () => {
    if (enrollmentPatterns.length < REQUIRED_PATTERNS) {
      setStatus({
        type: "error",
        message: `Need ${REQUIRED_PATTERNS} patterns for enrollment. You have ${enrollmentPatterns.length}.`,
      });
      return;
    }

    if (
      deviceTypeCounts[INPUT_DEVICE_TYPES.KEYROW] !== PATTERNS_PER_DEVICE ||
      deviceTypeCounts[INPUT_DEVICE_TYPES.NUMPAD] !== PATTERNS_PER_DEVICE
    ) {
      setStatus({
        type: "error",
        message: `Collect exactly ${PATTERNS_PER_DEVICE} patterns with the top-row numbers and ${PATTERNS_PER_DEVICE} with the numeric keypad before completing enrollment.`,
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
      setInputDeviceType(INPUT_DEVICE_TYPES.KEYROW);
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
              {user.username}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Current Input Device Required</label>
        <div
          style={{
            padding: "1rem",
            backgroundColor:
              inputDeviceType === INPUT_DEVICE_TYPES.NUMPAD
                ? "#e3f2fd"
                : "#fff3e0",
            borderRadius: "8px",
            border:
              "3px solid " +
              (inputDeviceType === INPUT_DEVICE_TYPES.NUMPAD
                ? "#2196f3"
                : "#ff9800"),
            fontWeight: "bold",
            textAlign: "center",
            fontSize: "1.1rem",
          }}
        >
          {inputDeviceType === INPUT_DEVICE_TYPES.NUMPAD
            ? "📱 USE NUMERIC KEYPAD ONLY"
            : "⌨️ USE TOP-ROW NUMBERS ONLY"}
        </div>
        <small
          style={{ color: "#4a5568", display: "block", marginTop: "0.5rem" }}
        >
          {getDeviceGuidance(inputDeviceType)} First {PATTERNS_PER_DEVICE}{" "}
          patterns require top-row, next {PATTERNS_PER_DEVICE} require numpad.
        </small>
      </div>

      <div className="form-group">
        <label htmlFor="testPhrase">Enrollment Phrase</label>
        <input
          type="text"
          id="testPhrase"
          value={typeof testPhrase === "string" ? testPhrase : testPhrase.text}
          readOnly
          placeholder="Enrollment phrase will randomize for each capture"
        />
        {testPhrase.source && (
          <small
            style={{
              color: "#6b7280",
              fontStyle: "italic",
              display: "block",
              marginTop: "0.25rem",
            }}
          >
            Source: {testPhrase.source}
          </small>
        )}
        <button
          type="button"
          className="btn btn-secondary"
          style={{ marginTop: "0.5rem" }}
          onClick={() => {
            stopCapture();
            setTypedText("");
            const currentText =
              typeof testPhrase === "string" ? testPhrase : testPhrase.text;
            setTestPhrase(getRandomPhrase(currentText));
            setStatus({
              type: "info",
              message: `Phrase updated. Use ${getDeviceTypeLabel(
                inputDeviceType
              ).toLowerCase()}. ${getDeviceGuidance(inputDeviceType)}`,
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
          onKeyDown={handleKeyDownWithValidation}
          onKeyUp={handleKeyUp}
          placeholder="Click 'Start Enrollment' or 'Start Next Pattern' then type the phrase above..."
          disabled={!isCapturing}
          onPaste={(e) => e.preventDefault()}
        />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <p>
          <strong>Target:</strong>{" "}
          {typeof testPhrase === "string" ? testPhrase : testPhrase.text}
        </p>
        <p>
          <strong>Typed:</strong> {typedText}
        </p>
        <p>
          <strong>Progress:</strong> {typedText.length}/
          {typeof testPhrase === "string"
            ? testPhrase.length
            : testPhrase.text.length}{" "}
          characters
        </p>
        <p>
          <strong>Patterns Collected:</strong> {enrollmentPatterns.length}/
          {REQUIRED_PATTERNS}
        </p>
        <p>
          <strong>Device Coverage:</strong> Top-row{" "}
          {deviceTypeCounts[INPUT_DEVICE_TYPES.KEYROW]}/{PATTERNS_PER_DEVICE} |
          Numpad {deviceTypeCounts[INPUT_DEVICE_TYPES.NUMPAD]}/
          {PATTERNS_PER_DEVICE}
        </p>
        {typedText ===
          (typeof testPhrase === "string" ? testPhrase : testPhrase.text) &&
          typedText.length > 0 && (
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
              isCapturing || enrollmentPatterns.length >= REQUIRED_PATTERNS
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
          disabled={loading || enrollmentPatterns.length < REQUIRED_PATTERNS}
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
                {REQUIRED_PATTERNS - enrollmentPatterns.length}
              </div>
              <div className="stat-label">Patterns Remaining</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {enrollmentPatterns.length >= REQUIRED_PATTERNS ? "✓" : "○"}
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
