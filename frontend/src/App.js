import React, { useState } from 'react';
import './App.css';
import UserRegistration from './components/UserRegistration';
import UserEnrollment from './components/UserEnrollment';
import UserAuthentication from './components/UserAuthentication';
import SystemStats from './components/SystemStats';

function App() {
  const [activeTab, setActiveTab] = useState('register');
  const [currentUser, setCurrentUser] = useState(null);

  const tabs = [
    { id: 'register', label: 'Register User', component: UserRegistration },
    { id: 'enroll', label: 'Enroll Pattern', component: UserEnrollment },
    { id: 'authenticate', label: 'Authenticate', component: UserAuthentication },
    { id: 'stats', label: 'System Stats', component: SystemStats },
  ];

  const renderActiveComponent = () => {
    const activeTabData = tabs.find(tab => tab.id === activeTab);
    if (!activeTabData) return null;
    
    const Component = activeTabData.component;
    return (
      <Component 
        currentUser={currentUser} 
        setCurrentUser={setCurrentUser}
      />
    );
  };

  const getTabStatus = (tabId) => {
    if (!currentUser) return '';
    
    switch (tabId) {
      case 'register':
        return '✓'; // User exists
      case 'enroll':
        return currentUser.is_enrolled ? '✓' : '⏳';
      case 'authenticate':
        return currentUser.is_enrolled ? '' : '🔒';
      default:
        return '';
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔐 Keystroke Dynamics Authentication</h1>
        <p>Biometric authentication based on typing patterns</p>
      </header>

      <nav className="App-nav">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`nav-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
            disabled={tab.id === 'authenticate' && currentUser && !currentUser.is_enrolled}
          >
            {tab.label} {getTabStatus(tab.id)}
          </button>
        ))}
      </nav>

      {currentUser && (
        <div className="user-status">
          <p>
            <strong>Current User:</strong> {currentUser.username} 
            {currentUser.is_enrolled ? 
              <span style={{color: 'green'}}> ✓ Enrolled</span> : 
              <span style={{color: 'orange'}}> ⏳ Need to complete enrollment</span>
            }
          </p>
        </div>
      )}

      <main className="App-main">
        {currentUser && (
          <div className="current-user">
            <p>Current User: <strong>{currentUser.username}</strong> (ID: {currentUser.id})</p>
          </div>
        )}
        {renderActiveComponent()}
      </main>

      <footer className="App-footer">
        <p>Keystroke Dynamics Authentication System | Real-time biometric security</p>
      </footer>
    </div>
  );
}

export default App;
