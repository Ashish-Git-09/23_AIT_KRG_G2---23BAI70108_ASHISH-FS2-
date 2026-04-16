package com.classroom.security.controller;

import com.classroom.security.model.AuthResponse;
import com.classroom.security.model.LoginRequest;
import com.classroom.security.model.RefreshRequest;
import com.classroom.security.util.JwtUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * ============================================================
 *  CHAPTER 3.1 — Authentication Controller
 * ============================================================
 *  POST /auth/login    — authenticates user, returns JWT pair
 *  POST /auth/refresh  — issues new access token via refresh token
 *  POST /auth/logout   — client-side only (stateless)
 *  GET  /auth/me       — returns current user info
 *
 *  These endpoints are permitAll() in SecurityConfig
 *  (except /auth/me which needs a valid token to identify the user)
 * ============================================================
 */
@Slf4j
@RestController
@RequestMapping("/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthenticationManager authManager;
    private final JwtUtil jwtUtil;

    // ── LOGIN ────────────────────────────────────────────────────────────────
    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest request) {
        log.debug("Login attempt for user: {}", request.getUsername());
        try {
            // Spring Security verifies username + password
            Authentication auth = authManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                    request.getUsername(), request.getPassword()
                )
            );

            // Extract roles from the authenticated principal
            List<String> roles = auth.getAuthorities().stream()
                    .map(GrantedAuthority::getAuthority)
                    .toList();

            // Generate both tokens
            String accessToken  = jwtUtil.generateAccessToken(auth.getName(), roles);
            String refreshToken = jwtUtil.generateRefreshToken(auth.getName());

            log.info("✓ Login successful: {} → roles: {}", auth.getName(), roles);

            return ResponseEntity.ok(new AuthResponse(
                accessToken, refreshToken, auth.getName(), roles
            ));

        } catch (Exception e) {
            log.warn("✗ Login failed for {}: {}", request.getUsername(), e.getMessage());
            return ResponseEntity.status(401)
                    .body(Map.of("error", "Invalid username or password"));
        }
    }

    // ── REFRESH TOKEN ────────────────────────────────────────────────────────
    @PostMapping("/refresh")
    public ResponseEntity<?> refresh(@RequestBody RefreshRequest request) {
        String refreshToken = request.getRefreshToken();
        log.debug("Refresh token request received");

        if (!jwtUtil.validateToken(refreshToken)) {
            log.warn("Invalid refresh token");
            return ResponseEntity.status(401)
                    .body(Map.of("error", "Refresh token is invalid or expired. Please login again."));
        }

        // Make sure this is actually a refresh token (not an access token being misused)
        if (!"refresh".equals(jwtUtil.extractType(refreshToken))) {
            return ResponseEntity.status(401)
                    .body(Map.of("error", "Invalid token type"));
        }

        String username = jwtUtil.extractUsername(refreshToken);
        // In a real app: look up roles from DB here
        // For demo: we re-issue with same roles (stored in original access token)
        List<String> roles = List.of("ROLE_USER"); // simplified for demo

        String newAccessToken = jwtUtil.generateAccessToken(username, roles);
        log.info("✓ Access token refreshed for: {}", username);

        return ResponseEntity.ok(Map.of(
            "accessToken", newAccessToken,
            "tokenType", "Bearer"
        ));
    }

    // ── LOGOUT (stateless — just confirmation) ───────────────────────────────
    @PostMapping("/logout")
    public ResponseEntity<?> logout() {
        // With stateless JWT, logout is client-side (delete token from storage)
        // In production: maintain a token blocklist in Redis
        return ResponseEntity.ok(Map.of(
            "message", "Logged out. Please delete your token on the client side."
        ));
    }
}
