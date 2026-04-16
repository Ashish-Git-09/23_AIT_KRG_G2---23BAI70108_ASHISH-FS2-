/**
 * ============================================================
 *  CHAPTER 3.2 — Axios Setup with Interceptors
 * ============================================================
 *  This file is the central API client for the frontend.
 *
 *  Concepts demonstrated:
 *  1. Single Axios instance with baseURL
 *  2. Request interceptor  → attaches JWT automatically
 *  3. Response interceptor → handles 401 / refresh token flow
 *  4. Error synchronization between frontend and backend
 * ============================================================
 */

// ── Create a configured Axios instance ──────────────────────────────────────
const api = axios.create({
  baseURL: 'http://localhost:8080',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 second timeout
});

// ── REQUEST INTERCEPTOR ──────────────────────────────────────────────────────
// Runs before every request is sent.
// Automatically attaches the JWT so every call doesn't have to do it manually.
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('accessToken');
    if (token) {
      // Chapter 3.2: Attaching JWT to headers
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;   // must return config to continue
  },
  (error) => {
    // Request setup failed (network issue before sending)
    return Promise.reject(error);
  }
);

// ── RESPONSE INTERCEPTOR ─────────────────────────────────────────────────────
// Runs after every response is received.
// Handles 401 Unauthorized globally — tries refresh before giving up.
api.interceptors.response.use(
  // ✓ Success path — pass through
  (response) => response,

  // ✗ Error path — check if we can recover
  async (error) => {
    const originalRequest = error.config;

    // Chapter 3.2: Handling 401 Unauthorized
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true; // prevent infinite retry loop

      const refreshToken = localStorage.getItem('refreshToken');

      if (refreshToken) {
        try {
          // Chapter 3.2: Refresh Token Flow
          const refreshResponse = await axios.post(
            'http://localhost:8080/auth/refresh',
            { refreshToken }
          );

          const newAccessToken = refreshResponse.data.accessToken;
          localStorage.setItem('accessToken', newAccessToken);

          // Update the failed request's header and retry it
          originalRequest.headers['Authorization'] = `Bearer ${newAccessToken}`;
          return api(originalRequest); // retry with new token

        } catch (refreshError) {
          // Refresh token also expired or invalid — force logout
          localStorage.removeItem('accessToken');
          localStorage.removeItem('refreshToken');
          localStorage.removeItem('currentUser');
          // In a React app: navigate('/login') or dispatch(logout())
          console.warn('Session expired. Please log in again.');
        }
      }
    }

    // Chapter 3.2: Error synchronization — surface backend error messages
    if (error.response?.data?.message) {
      error.userMessage = error.response.data.message;
    } else if (error.response?.status === 403) {
      error.userMessage = "You don't have permission to access this.";
    } else if (!error.response) {
      error.userMessage = "Cannot connect to server. Is the backend running?";
    }

    return Promise.reject(error);
  }
);

// Export (works as a global in plain HTML, or export default api in React)
// In React/Vite: export default api;
