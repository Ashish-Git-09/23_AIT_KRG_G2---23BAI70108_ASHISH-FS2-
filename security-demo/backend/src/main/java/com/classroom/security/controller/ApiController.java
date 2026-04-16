package com.classroom.security.controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * ============================================================
 *  CHAPTER 3.1 — RBAC with @PreAuthorize
 * ============================================================
 *  Demonstrates Role-Based Access Control.
 *
 *  @PreAuthorize runs AFTER JwtAuthFilter sets the Authentication
 *  in SecurityContextHolder. It checks the roles in the JWT.
 *
 *  Try these endpoints with different users:
 *    alice  (ADMIN)     — can access all
 *    bob    (USER)      — /user only
 *    carol  (MODERATOR) — /user + /moderator
 *    no JWT             — 401 on everything
 * ============================================================
 */
@Slf4j
@RestController
public class ApiController {

    // ── PUBLIC — no JWT needed ───────────────────────────────────────────────
    @GetMapping("/public/hello")
    public ResponseEntity<?> publicHello() {
        return ResponseEntity.ok(Map.of(
            "message", "Hello! This endpoint is public — no JWT needed.",
            "time", LocalDateTime.now().toString()
        ));
    }

    // ── AUTHENTICATED — any valid JWT ────────────────────────────────────────
    @GetMapping("/api/profile")
    public ResponseEntity<?> profile(Authentication auth) {
        log.debug("Profile requested by: {}", auth.getName());
        return ResponseEntity.ok(Map.of(
            "username", auth.getName(),
            "roles", auth.getAuthorities().stream()
                         .map(Object::toString).toList(),
            "message", "Your profile — any authenticated user can see this"
        ));
    }

    // ── USER role ────────────────────────────────────────────────────────────
    @GetMapping("/api/user/dashboard")
    @PreAuthorize("hasAnyRole('USER', 'ADMIN', 'MODERATOR')")
    public ResponseEntity<?> userDashboard(Authentication auth) {
        return ResponseEntity.ok(Map.of(
            "message", "Welcome to the User Dashboard, " + auth.getName() + "!",
            "data", List.of("Item 1", "Item 2", "Item 3"),
            "accessLevel", "USER"
        ));
    }

    // ── MODERATOR role ───────────────────────────────────────────────────────
    @GetMapping("/api/moderator/posts")
    @PreAuthorize("hasAnyRole('MODERATOR', 'ADMIN')")
    public ResponseEntity<?> moderatorPosts(Authentication auth) {
        return ResponseEntity.ok(Map.of(
            "message", "Moderator view — manage posts",
            "posts", List.of(
                Map.of("id", 1, "title", "Post A", "status", "flagged"),
                Map.of("id", 2, "title", "Post B", "status", "approved")
            ),
            "accessLevel", "MODERATOR"
        ));
    }

    // ── ADMIN role only ──────────────────────────────────────────────────────
    @GetMapping("/api/admin/users")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<?> adminUsers() {
        // Only alice can reach this
        return ResponseEntity.ok(Map.of(
            "message", "Admin panel — all users",
            "users", List.of(
                Map.of("username", "alice", "role", "ADMIN"),
                Map.of("username", "bob",   "role", "USER"),
                Map.of("username", "carol", "role", "MODERATOR")
            ),
            "accessLevel", "ADMIN"
        ));
    }

    @DeleteMapping("/api/admin/users/{username}")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<?> deleteUser(@PathVariable String username, Authentication auth) {
        log.info("Admin {} attempting to delete user: {}", auth.getName(), username);
        return ResponseEntity.ok(Map.of(
            "message", "User '" + username + "' deleted by " + auth.getName(),
            "note", "This is a demo — no actual deletion happens!"
        ));
    }
}
