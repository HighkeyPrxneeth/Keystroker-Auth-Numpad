import React, { useState, useEffect } from 'react';
import apiService from '../services/api';

const SystemStats = () => {
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState('');
  const [userPerformance, setUserPerformance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    loadSystemStats();
    loadUsers();
  }, []);

  const loadSystemStats = async () => {
    try {
      setLoading(true);
      const data = await apiService.getSystemStats();
      setStats(data);
    } catch (err) {
      setError('Failed to load system statistics');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadUsers = async () => {
    try {
      const userList = await apiService.getAllUsers();
      setUsers(userList);
    } catch (err) {
      console.error('Failed to load users:', err);
    }
  };

  const handleUserSelect = async (userId) => {
    setSelectedUser(userId);
    if (!userId) {
      setUserPerformance(null);
      return;
    }

    try {
      setLoading(true);
      const performance = await apiService.getUserPerformance(userId);
      setUserPerformance(performance);
    } catch (err) {
      setError('Failed to load user performance data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const refreshStats = () => {
    loadSystemStats();
    if (selectedUser) {
      handleUserSelect(selectedUser);
    }
  };

  if (loading && !stats) {
    return (
      <div className="card">
        <h2>📊 System Statistics</h2>
        <p>Loading system statistics...</p>
      </div>
    );
  }

  return (
    <div>
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2>📊 System Statistics</h2>
          <button className="btn btn-secondary" onClick={refreshStats}>
            Refresh Data
          </button>
        </div>
        
        {error && (
          <div className="status-message status-error">
            {error}
          </div>
        )}

        {stats && (
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{stats.total_users}</div>
              <div className="stat-label">Total Users</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.total_patterns}</div>
              <div className="stat-label">Enrolled Patterns</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.total_authentications}</div>
              <div className="stat-label">Authentication Attempts</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.successful_authentications}</div>
              <div className="stat-label">Successful Authentications</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {stats.total_authentications > 0 
                  ? ((stats.successful_authentications / stats.total_authentications) * 100).toFixed(1) + '%'
                  : '0%'
                }
              </div>
              <div className="stat-label">Success Rate</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.active_models}</div>
              <div className="stat-label">Active ML Models</div>
            </div>
          </div>
        )}

        {stats?.recent_activity && stats.recent_activity.length > 0 && (
          <div style={{ marginTop: '2rem' }}>
            <h3>Recent Activity</h3>
            <div style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '8px', maxHeight: '200px', overflowY: 'auto' }}>
              {stats.recent_activity.map((activity, index) => (
                <div key={index} style={{ marginBottom: '0.5rem', padding: '0.5rem', background: 'white', borderRadius: '4px' }}>
                  <strong>{activity.type}:</strong> {activity.description}
                  <div style={{ fontSize: '0.8rem', color: '#666' }}>
                    {new Date(activity.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h3>👤 User Performance Analysis</h3>
        
        <div className="form-group">
          <label htmlFor="userSelect">Select User</label>
          <select
            id="userSelect"
            value={selectedUser}
            onChange={(e) => handleUserSelect(e.target.value)}
            style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '2px solid #e1e5e9' }}
          >
            <option value="">Choose a user to analyze...</option>
            {users.map(user => (
              <option key={user.id} value={user.id}>
                {user.username} ({user.email})
              </option>
            ))}
          </select>
        </div>

        {userPerformance && (
          <div>
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">{userPerformance.enrolled_patterns}</div>
                <div className="stat-label">Enrolled Patterns</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{userPerformance.authentication_attempts}</div>
                <div className="stat-label">Auth Attempts</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{userPerformance.successful_attempts}</div>
                <div className="stat-label">Successful</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">
                  {userPerformance.authentication_attempts > 0 
                    ? ((userPerformance.successful_attempts / userPerformance.authentication_attempts) * 100).toFixed(1) + '%'
                    : '0%'
                  }
                </div>
                <div className="stat-label">Success Rate</div>
              </div>
            </div>

            {userPerformance.performance_metrics && (
              <div style={{ marginTop: '2rem' }}>
                <h4>Performance Metrics</h4>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-value">
                      {userPerformance.performance_metrics.avg_typing_speed?.toFixed(1) || 'N/A'}
                    </div>
                    <div className="stat-label">Avg Typing Speed (WPM)</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">
                      {userPerformance.performance_metrics.avg_dwell_time?.toFixed(1) || 'N/A'}
                    </div>
                    <div className="stat-label">Avg Dwell Time (ms)</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">
                      {userPerformance.performance_metrics.avg_flight_time?.toFixed(1) || 'N/A'}
                    </div>
                    <div className="stat-label">Avg Flight Time (ms)</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">
                      {userPerformance.performance_metrics.consistency_score?.toFixed(2) || 'N/A'}
                    </div>
                    <div className="stat-label">Consistency Score</div>
                  </div>
                </div>
              </div>
            )}

            {userPerformance.recent_attempts && userPerformance.recent_attempts.length > 0 && (
              <div style={{ marginTop: '2rem' }}>
                <h4>Recent Authentication Attempts</h4>
                <div style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '8px', maxHeight: '200px', overflowY: 'auto' }}>
                  {userPerformance.recent_attempts.map((attempt, index) => (
                    <div key={index} style={{ 
                      marginBottom: '0.5rem', 
                      padding: '0.5rem', 
                      background: attempt.success ? '#d4edda' : '#f8d7da', 
                      borderRadius: '4px' 
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>
                          <strong>{attempt.success ? '✓' : '✗'}</strong> 
                          Confidence: {(attempt.confidence * 100).toFixed(1)}%
                        </span>
                        <span style={{ fontSize: '0.8rem' }}>
                          {new Date(attempt.timestamp).toLocaleString()}
                        </span>
                      </div>
                      {attempt.typing_speed && (
                        <div style={{ fontSize: '0.8rem', color: '#666' }}>
                          Typing Speed: {attempt.typing_speed.toFixed(1)} WPM
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SystemStats;
