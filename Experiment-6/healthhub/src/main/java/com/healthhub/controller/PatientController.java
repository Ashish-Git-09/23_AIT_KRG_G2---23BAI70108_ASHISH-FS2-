package com.healthhub.controller;

import com.healthhub.dto.PatientDTO;
import com.healthhub.service.PatientService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * PatientController — REST Controller Layer.
 * @RestController = @Controller + @ResponseBody (returns JSON automatically).
 * @RequestMapping sets the base path for all endpoints.
 * Service is INJECTED via constructor (Dependency Injection).
 */
@Slf4j
@RestController
@RequestMapping("/api/patients")
@RequiredArgsConstructor
public class PatientController {

    private final PatientService patientService;

    // ── POST /api/patients — Create ───────────────────────────────────────────
    @PostMapping
    public ResponseEntity<PatientDTO> createPatient(@Valid @RequestBody PatientDTO dto) {
        log.debug("POST /api/patients called");
        PatientDTO created = patientService.createPatient(dto);
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }

    // ── GET /api/patients — Read All ──────────────────────────────────────────
    @GetMapping
    public ResponseEntity<List<PatientDTO>> getAllPatients() {
        log.debug("GET /api/patients called");
        return ResponseEntity.ok(patientService.getAllPatients());
    }

    // ── GET /api/patients/{id} — Read One ─────────────────────────────────────
    @GetMapping("/{id}")
    public ResponseEntity<PatientDTO> getPatientById(@PathVariable Long id) {
        log.debug("GET /api/patients/{} called", id);
        return ResponseEntity.ok(patientService.getPatientById(id));
    }

    // ── PUT /api/patients/{id} — Update ──────────────────────────────────────
    @PutMapping("/{id}")
    public ResponseEntity<PatientDTO> updatePatient(@PathVariable Long id,
                                                     @Valid @RequestBody PatientDTO dto) {
        log.debug("PUT /api/patients/{} called", id);
        return ResponseEntity.ok(patientService.updatePatient(id, dto));
    }

    // ── DELETE /api/patients/{id} — Delete ───────────────────────────────────
    @DeleteMapping("/{id}")
    public ResponseEntity<Map<String, String>> deletePatient(@PathVariable Long id) {
        log.debug("DELETE /api/patients/{} called", id);
        patientService.deletePatient(id);
        return ResponseEntity.ok(Map.of("message", "Patient with ID " + id + " deleted successfully."));
    }
}
