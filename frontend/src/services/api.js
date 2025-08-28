const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

class ApiService {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}/api${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    if (config.body && typeof config.body !== 'string') {
      config.body = JSON.stringify(config.body);
    }

    try {
      const response = await fetch(url, config);
      
      let data;
      try {
        data = await response.json();
      } catch (jsonError) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }
      
      if (!response.ok) {
        const errorMessage = data.detail || data.message || `HTTP ${response.status}: ${response.statusText}`;
        throw new Error(errorMessage);
      }
      
      return data;
    } catch (error) {
      console.error('API Error:', error);
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error('Cannot connect to server. Make sure the backend is running.');
      }
      throw error;
    }
  }

  // User management
  async createUser(userData) {
    return this.request('/users', {
      method: 'POST',
      body: userData,
    });
  }

  async getUser(userId) {
    return this.request(`/users/${userId}`);
  }

  async getAllUsers() {
    return this.request('/users');
  }

  // Keystroke pattern collection
  async collectPattern(patternData) {
    return this.request('/collect-pattern', {
      method: 'POST',
      body: patternData,
    });
  }

  // Enrollment
  async enrollUser(enrollmentData) {
    return this.request('/enroll', {
      method: 'POST',
      body: enrollmentData,
    });
  }

  // Authentication
  async authenticateUser(authData) {
    return this.request('/authenticate', {
      method: 'POST',
      body: authData,
    });
  }

  // System stats
  async getSystemStats() {
    return this.request('/system/stats');
  }

  async getUserPerformance(userId) {
    return this.request(`/users/${userId}/performance`);
  }

  // Health check
  async healthCheck() {
    return this.request('/health');
  }
}

export default new ApiService();
