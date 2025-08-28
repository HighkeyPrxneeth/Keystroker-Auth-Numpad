import React, { useState, useEffect } from 'react';
import { useKeystrokeCapture } from '../hooks/useKeystrokeCapture';
import apiService from '../services/api';

const UserAuthentication = () => {
  const [users, setUsers] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState('');
  const [testPhrase, setTestPhrase] = useState('The quick brown fox jumps over the lazy dog');
  const [typedText, setTypedText] = useState('');
  const [status, setStatus] = useState({ type: '', message: '' });
  const [loading, setLoading] = useState(false);
  const [authResult, setAuthResult] = useState(null);

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
      console.error('Failed to load users:', error);
    }
  };

  const handleStartAuth = () => {
    setTypedText('');
    setStatus({ type: '', message: '' });
    setAuthResult(null);
    startCapture();
    setStatus({
      type: 'info',
      message: 'Recording keystroke pattern... Type the phrase below.',
    });
  };

  const handleTextChange = (e) => {
    const value = e.target.value;
    setTypedText(value);

    // Check if user completed the phrase
    if (value === testPhrase && isCapturing) {
      stopCapture();
      setStatus({
        type: 'info',
        message: 'Pattern captured! Click "Authenticate" to verify identity.',
      });
    }
  };

  const handleAuthenticate = async () => {
    if (!selectedUserId) {
      setStatus({
        type: 'error',
        message: 'Please select a user to authenticate.',
      });
      return;
    }

    if (keystrokeData.length === 0) {
      setStatus({
        type: 'error',
        message: 'No keystroke data captured. Please type the phrase first.',
      });
      return;
    }

    setLoading(true);
    try {
      // Find the selected user to get their username
      const selectedUser = users.find(user => user.id === selectedUserId);
      if (!selectedUser) {
        setStatus({
          type: 'error',
          message: 'Selected user not found.',
        });
        setLoading(false);
        return;
      }

      const pattern = getTypingPattern();
      
      // Convert keystroke data to the format expected by backend
      // The backend expects keydown and keyup events, but our hook only captures keyup
      // So we'll simulate both events with proper timing
      const events = [];
      pattern.keystrokes.forEach(keystroke => {
        // Add keydown event (before the keyup)
        events.push({
          key: keystroke.key,
          key_code: keystroke.keyCode || 0,
          timestamp: keystroke.timestamp - keystroke.dwellTime, // Start of key press
          event_type: 'keydown'
        });
        
        // Add keyup event
        events.push({
          key: keystroke.key,
          key_code: keystroke.keyCode || 0,
          timestamp: keystroke.timestamp, // End of key press
          event_type: 'keyup'
        });
      });

      const authData = {
        username: selectedUser.username,
        pattern: {
          text_typed: typedText,
          events: events
        }
      };

      const result = await apiService.authenticateUser(authData);
      setAuthResult(result);
      
      // The backend returns status: "success" or "failed" and confidence_score
      const isAuthenticated = result.status === 'success';
      const confidence = result.confidence_score || 0;
      
      if (isAuthenticated) {
        setStatus({
          type: 'success',
          message: `Authentication successful! Confidence: ${(confidence * 100).toFixed(1)}%`,
        });
      } else {
        setStatus({
          type: 'error',
          message: `Authentication failed. Confidence: ${(confidence * 100).toFixed(1)}%`,
        });
      }
      
      setTypedText('');
    } catch (error) {
      let errorMessage = error.message || 'Authentication failed';
      
      // Check for specific enrollment error
      if (errorMessage.includes('User not enrolled') || errorMessage.includes('not enrolled')) {
        errorMessage = 'User has not completed enrollment yet. Please complete the enrollment process first.';
      }
      
      setStatus({
        type: 'error',
        message: errorMessage,
      });
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#28a745';
    if (confidence >= 0.6) return '#ffc107';
    return '#dc3545';
  };

  return (
    <div className="card">
      <h2>🔐 User Authentication</h2>
      <p>Type the phrase to authenticate using your unique keystroke pattern.</p>

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
          style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '2px solid #e1e5e9' }}
        >
          <option value="">Choose a user...</option>
          {users.map(user => (
            <option key={user.id} value={user.id}>
              {user.username} ({user.email})
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="testPhrase">Authentication Phrase</label>
        <input
          type="text"
          id="testPhrase"
          value={testPhrase}
          onChange={(e) => setTestPhrase(e.target.value)}
          placeholder="Enter the phrase to type"
        />
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

      <div style={{ marginBottom: '1rem' }}>
        <p><strong>Target:</strong> {testPhrase}</p>
        <p><strong>Typed:</strong> {typedText}</p>
        <p><strong>Progress:</strong> {typedText.length}/{testPhrase.length} characters</p>
        {typedText === testPhrase && typedText.length > 0 && (
          <p style={{ color: 'green', fontWeight: 'bold' }}>✓ Phrase completed!</p>
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
          {loading ? 'Authenticating...' : 'Authenticate'}
        </button>
      </div>

      {authResult && (
        <div className="card" style={{ marginTop: '2rem', backgroundColor: authResult.status === 'success' ? '#d4edda' : '#f8d7da' }}>
          <h3>Authentication Result</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value" style={{ color: authResult.status === 'success' ? '#28a745' : '#dc3545' }}>
                {authResult.status === 'success' ? '✓' : '✗'}
              </div>
              <div className="stat-label">Status</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: getConfidenceColor(authResult.confidence_score || 0) }}>
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
                {authResult.message || 'No message'}
              </div>
              <div className="stat-label">Details</div>
            </div>
          </div>
        </div>
      )}

      {keystrokeData.length > 0 && (
        <div className="keystroke-display">
          <h4>Captured Keystrokes: {keystrokeData.length}</h4>
          <div style={{ maxHeight: '150px', overflowY: 'auto' }}>
            {keystrokeData.map((keystroke, index) => (
              <div key={index}>
                Key: {keystroke.key} | Dwell: {keystroke.dwellTime}ms | Time: {keystroke.timestamp}ms
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default UserAuthentication;
