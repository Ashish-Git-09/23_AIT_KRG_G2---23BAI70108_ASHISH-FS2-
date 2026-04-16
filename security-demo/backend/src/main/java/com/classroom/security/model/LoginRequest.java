package com.classroom.security.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

// ── Request DTOs ─────────────────────────────────────────────────────────────

/**
 * Login request body: { "username": "alice", "password": "password123" }
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class LoginRequest {
    private String username;
    private String password;
}
