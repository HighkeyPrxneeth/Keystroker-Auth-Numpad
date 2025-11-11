import React, { useState } from "react";
import apiService from "../services/api";

const UserRegistration = ({ setCurrentUser }) => {
  const [formData, setFormData] = useState({
    username: "",
  });
  const [status, setStatus] = useState({ type: "", message: "" });
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setStatus({ type: "", message: "" });

    try {
      const userData = {
        username: formData.username,
        email: `${formData.username}@keystroke.local`, // Auto-generate email
      };

      const response = await apiService.createUser(userData);
      setStatus({
        type: "success",
        message: `User registered successfully! User ID: ${response.id}`,
      });
      setCurrentUser(response);
      setFormData({ username: "" });
    } catch (error) {
      console.error("Registration error:", error);
      setStatus({
        type: "error",
        message:
          typeof error === "string"
            ? error
            : error.message || "Failed to register user",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>👤 Register New User</h2>
      <p>Create a new user account for keystroke pattern enrollment.</p>

      {status.message && (
        <div className={`status-message status-${status.type}`}>
          {status.message}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="username">Username</label>
          <input
            type="text"
            id="username"
            name="username"
            value={formData.username}
            onChange={handleChange}
            required
            placeholder="Enter your username"
          />
        </div>

        <button type="submit" className="btn" disabled={loading}>
          {loading ? "Registering..." : "Register User"}
        </button>
      </form>
    </div>
  );
};

export default UserRegistration;
