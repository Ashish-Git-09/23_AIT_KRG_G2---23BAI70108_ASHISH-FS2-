# 🔐 Security Demo — Chapter 3.1 & 3.2
### Application Security + Full Stack Integration

A complete classroom demonstration project for teaching:
- Security Filter Chain, JWT (generation/signing/validation), RBAC with `@PreAuthorize`
- CORS configuration, Axios interceptors, Refresh token flow

---

## 📁 Project Structure

```
security-demo/
├── backend/                        ← Spring Boot project (open this in IntelliJ)
│   ├── pom.xml
│   └── src/main/java/com/classroom/security/
│       ├── SecurityDemoApplication.java    ← Main class — run this!
│       ├── config/
│       │   ├── SecurityConfig.java         ← Filter chain + CORS + users
│       │   └── GlobalExceptionHandler.java ← 401/403 error responses
│       ├── controller/
│       │   ├── AuthController.java         ← /auth/login, /auth/refresh
│       │   └── ApiController.java          ← RBAC-protected endpoints
│       ├── filter/
│       │   └── JwtAuthFilter.java          ← Reads + validates JWT per request
│       ├── model/
│       │   ├── LoginRequest.java
│       │   ├── AuthResponse.java
│       │   └── RefreshRequest.java
│       └── util/
│           └── JwtUtil.java                ← Token generation, signing, validation
└── frontend/
    ├── index.html                          ← Open in browser after starting backend
    └── api.js                             ← Axios instance + interceptors
```

---

## 🚀 Setup & Run (5 Minutes)

### Prerequisites
- Java 17+
- Maven 3.8+ (or use the IntelliJ Maven wrapper)
- A modern browser (Chrome/Firefox)

### Step 1 — Open in IntelliJ
1. Open IntelliJ IDEA
2. **File → Open** → select the `backend/` folder
3. IntelliJ will detect the `pom.xml` — click **"Load Maven Project"** if prompted
4. Wait for Maven to download dependencies (~30 seconds on first run)

### Step 2 — Run the Backend
1. Navigate to `src/main/java/com/classroom/security/SecurityDemoApplication.java`
2. Click the **▶ green Run button** next to the `main` method
3. Watch the console — you should see:
   ```
   ╔══════════════════════════════════════════╗
   ║   Security Demo is running!              ║
   ║   API Base: http://localhost:8080        ║
   ╚══════════════════════════════════════════╝
   ```

### Step 3 — Open the Frontend
1. Navigate to the `frontend/` folder
2. Open `index.html` directly in Chrome or Firefox
   - **Double-click** the file, or
   - Drag it into the browser
3. The status bar should show **"Backend online :8080"** (green dot)

---

## 👥 Test Users (in-memory, no database)

| Username | Password    | Role           |
|----------|-------------|----------------|
| alice    | password123 | ROLE_ADMIN     |
| bob      | password123 | ROLE_USER      |
| carol    | password123 | ROLE_MODERATOR |

---

## 🎓 Classroom Demo Script

### Demo 1 — Authentication vs Authorization (Chapter 3.1)
1. Show `SecurityConfig.java` — explain the filter chain order
2. Show `JwtAuthFilter.java` — walk through the 5 steps in the comments
3. In the UI: click **alice** in the sidebar → login → show the JWT response
4. Show `JwtUtil.java` — explain Header.Payload.Signature structure

### Demo 2 — JWT Inspector
1. Go to **JWT Inspector** tab in the UI
2. Click "Load current access token"
3. Students can see the decoded Header and Payload live
4. Point out: `sub` (username), `roles`, `exp` (expiry timestamp)
5. **Try tampering**: change one character of the token → show Spring rejects it

### Demo 3 — RBAC with @PreAuthorize
1. Show `ApiController.java` — the 4 endpoint levels
2. Log in as **bob** (USER) → try the ADMIN endpoint → get 403
3. Log in as **alice** (ADMIN) → same endpoint → 200 OK
4. Explain: `@PreAuthorize` checks `SecurityContextHolder` which `JwtAuthFilter` populated

### Demo 4 — CORS (Chapter 3.2)
1. Go to **CORS Demo** tab → click the button
2. Open **DevTools → Network tab** → find the request
3. Show students the `Access-Control-Allow-Origin` response header
4. Show `SecurityConfig.corsConfigurationSource()` — explain each line

### Demo 5 — Axios Interceptors
1. Go to **Axios Interceptors** tab → open `frontend/api.js` side by side
2. Click "Request WITH valid JWT" → show the Authorization header was auto-added
3. Click "No token" → show 401
4. Click "Tampered token" → show Spring's signature validation rejecting it

### Demo 6 — Refresh Token Flow
1. Go to **Refresh Token Flow** tab
2. Click "Simulate: Access Token Expired → Refresh → Retry"
3. Watch the animated step-by-step flow
4. Click "POST /auth/refresh" to actually test the endpoint live

---

## 🔌 API Endpoints Reference

| Method | URL                        | Auth Required | Roles          |
|--------|----------------------------|---------------|----------------|
| POST   | /auth/login                | No            | —              |
| POST   | /auth/refresh              | No            | —              |
| POST   | /auth/logout               | No            | —              |
| GET    | /public/hello              | No            | —              |
| GET    | /api/profile               | Yes (JWT)     | Any            |
| GET    | /api/user/dashboard        | Yes           | USER, MOD, ADMIN |
| GET    | /api/moderator/posts       | Yes           | MOD, ADMIN     |
| GET    | /api/admin/users           | Yes           | ADMIN only     |
| DELETE | /api/admin/users/{username}| Yes           | ADMIN only     |

---

## 🧪 Quick Test with curl (for showing in terminal)

```bash
# 1. Login
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"password123"}'

# 2. Copy the accessToken, then:
TOKEN="paste_token_here"

# 3. Call protected endpoint
curl http://localhost:8080/api/admin/users \
  -H "Authorization: Bearer $TOKEN"

# 4. Try with bob's token — expect 403
```

---

## ⚠️ Common Issues

| Problem | Solution |
|---------|----------|
| "Cannot connect to server" | Start `SecurityDemoApplication.java` first |
| Maven dependencies not downloading | Check internet connection; File → Invalidate Caches |
| Port 8080 already in use | Stop other Spring apps, or change `server.port` in `application.yml` |
| CORS error in browser console | Use Chrome/Firefox (not Safari file://); check `corsConfigurationSource()` includes `"null"` origin |
| 401 on all requests | Your token expired — click logout then login again |
