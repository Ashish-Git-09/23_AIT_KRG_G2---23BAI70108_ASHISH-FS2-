package com.healthhub.controller;

import com.healthhub.entity.Appointment;
import com.healthhub.entity.Doctor;
import com.healthhub.service.DoctorService;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

/**
 * DoctorController — exposes endpoints to demo all JPA & performance features.
 */
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class DoctorController {

    private final DoctorService doctorService;

    // ── Basic CRUD ────────────────────────────────────────────────────────────
    @PostMapping("/doctors")
    public ResponseEntity<Doctor> createDoctor(@RequestBody Doctor doctor) {
        return ResponseEntity.ok(doctorService.createDoctor(doctor));
    }

    @GetMapping("/doctors")
    public ResponseEntity<List<Doctor>> getAllDoctors() {
        return ResponseEntity.ok(doctorService.getAllDoctors());
    }

    @GetMapping("/doctors/{id}")
    public ResponseEntity<?> getDoctorById(@PathVariable Long id) {
        return doctorService.getDoctorById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    // ── JPQL Query endpoints ──────────────────────────────────────────────────
    @GetMapping("/doctors/specialization/{spec}")
    public ResponseEntity<List<Doctor>> bySpecialization(@PathVariable String spec) {
        return ResponseEntity.ok(doctorService.getDoctorsBySpecialization(spec));
    }

    @GetMapping("/doctors/search")
    public ResponseEntity<List<Doctor>> searchDoctors(@RequestParam String name) {
        return ResponseEntity.ok(doctorService.searchDoctors(name));
    }

    @GetMapping("/doctors/experienced")
    public ResponseEntity<List<Doctor>> experienced(@RequestParam(defaultValue = "5") int minYears) {
        return ResponseEntity.ok(doctorService.getExperiencedDoctors(minYears));
    }

    // ── Pagination ────────────────────────────────────────────────────────────
    @GetMapping("/doctors/paginated")
    public ResponseEntity<Map<String, Object>> paginated(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "3") int size,
            @RequestParam(defaultValue = "name") String sortBy) {
        return ResponseEntity.ok(doctorService.getDoctorsPaginated(page, size, sortBy));
    }

    // ── N+1 Demo endpoints ────────────────────────────────────────────────────
    @GetMapping("/doctors/with-appointments/naive")
    public ResponseEntity<List<Map<String, Object>>> naiveFetch() {
        return ResponseEntity.ok(doctorService.getDoctorsWithAppointmentsNaive());
    }

    @GetMapping("/doctors/with-appointments/optimized")
    public ResponseEntity<List<Map<String, Object>>> optimizedFetch() {
        return ResponseEntity.ok(doctorService.getDoctorsWithAppointmentsOptimized());
    }

    // ── Appointment endpoints ─────────────────────────────────────────────────
    @GetMapping("/appointments/status/{status}")
    public ResponseEntity<List<Appointment>> byStatus(@PathVariable String status) {
        return ResponseEntity.ok(doctorService.getAppointmentsByStatus(status));
    }

    @GetMapping("/appointments/daterange")
    public ResponseEntity<List<Appointment>> byDateRange(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate start,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate end) {
        return ResponseEntity.ok(doctorService.getAppointmentsByDateRange(start, end));
    }

    @GetMapping("/appointments/count-per-doctor")
    public ResponseEntity<List<Map<String, Object>>> countPerDoctor() {
        List<Object[]> raw = doctorService.getAppointmentCountPerDoctor();
        List<Map<String, Object>> result = raw.stream()
                .map(row -> Map.of("doctor", row[0], "count", row[1]))
                .toList();
        return ResponseEntity.ok(result);
    }

    @GetMapping("/appointments/paginated")
    public ResponseEntity<Map<String, Object>> appointmentsPaginated(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "5") int size) {
        return ResponseEntity.ok(doctorService.getAppointmentsPaginated(page, size));
    }
}
