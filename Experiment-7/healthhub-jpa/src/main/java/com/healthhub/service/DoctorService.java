package com.healthhub.service;

import com.healthhub.entity.Appointment;
import com.healthhub.entity.Doctor;
import com.healthhub.repository.AppointmentRepository;
import com.healthhub.repository.DoctorRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.util.*;

@Slf4j
@Service
@RequiredArgsConstructor
public class DoctorService {

    private final DoctorRepository doctorRepository;
    private final AppointmentRepository appointmentRepository;

    // ── CRUD ──────────────────────────────────────────────────────────────────
    public Doctor createDoctor(Doctor doctor) {
        return doctorRepository.save(doctor);
    }

    public Optional<Doctor> getDoctorById(Long id) {
        return doctorRepository.findById(id);
    }

    public List<Doctor> getAllDoctors() {
        return doctorRepository.findAll();
    }

    // ── JPQL Queries ──────────────────────────────────────────────────────────
    public List<Doctor> getDoctorsBySpecialization(String spec) {
        log.info("JPQL query: findBySpecialization({})", spec);
        return doctorRepository.findBySpecialization(spec);
    }

    public List<Doctor> searchDoctors(String name) {
        log.info("JPQL query: searchByName({})", name);
        return doctorRepository.searchByName(name);
    }

    public List<Doctor> getExperiencedDoctors(int minYears) {
        log.info("JPQL query: findExperiencedDoctors(minYears={})", minYears);
        return doctorRepository.findExperiencedDoctors(minYears);
    }

    // ── Pagination ────────────────────────────────────────────────────────────
    /**
     * Cursor-based pagination.
     * page=0 → first page, page=1 → second page, etc.
     * size=5 → 5 records per page.
     * sort=name → sorted alphabetically by name.
     *
     * Hibernate generates: SELECT * FROM doctors ORDER BY name LIMIT 5 OFFSET 0
     */
    public Map<String, Object> getDoctorsPaginated(int page, int size, String sortBy) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(sortBy));
        Page<Doctor> pageResult = doctorRepository.findAll(pageable);

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("doctors", pageResult.getContent());
        response.put("currentPage", pageResult.getNumber());
        response.put("totalItems", pageResult.getTotalElements());
        response.put("totalPages", pageResult.getTotalPages());
        response.put("hasNext", pageResult.hasNext());
        response.put("hasPrevious", pageResult.hasPrevious());
        return response;
    }

    // ── N+1 Problem DEMO ──────────────────────────────────────────────────────
    /**
     * BAD approach — triggers N+1 problem.
     * 1 query to get all doctors + N queries (one per doctor) to get their appointments.
     * Watch the SQL logs when this runs!
     */
    @Transactional
    public List<Map<String, Object>> getDoctorsWithAppointmentsNaive() {
        log.warn("=== N+1 DEMO: Naive fetch — watch how many SQL queries fire! ===");
        List<Doctor> doctors = doctorRepository.findAll(); // Query 1

        List<Map<String, Object>> result = new ArrayList<>();
        for (Doctor d : doctors) {
            // Each call to d.getAppointments() fires a NEW SQL query! (N queries)
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("doctor", d.getName());
            entry.put("appointmentCount", d.getAppointments().size()); // triggers SELECT
            result.add(entry);
        }
        log.warn("=== N+1 DEMO: Done. Total queries fired = 1 + {} ===", doctors.size());
        return result;
    }

    /**
     * GOOD approach — JOIN FETCH eliminates N+1.
     * Single SQL query with LEFT JOIN fetches doctors + all appointments together.
     */
    @Transactional
    public List<Map<String, Object>> getDoctorsWithAppointmentsOptimized() {
        log.info("=== OPTIMIZED: JOIN FETCH — only 1 SQL query fires! ===");
        List<Doctor> doctors = doctorRepository.findAllWithAppointments(); // 1 query only

        List<Map<String, Object>> result = new ArrayList<>();
        for (Doctor d : doctors) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("doctor", d.getName());
            entry.put("specialization", d.getSpecialization());
            entry.put("appointmentCount", d.getAppointments().size()); // already loaded!
            entry.put("appointments", d.getAppointments().stream()
                    .map(a -> Map.of(
                        "id", a.getId(),
                        "patient", a.getPatientName(),
                        "date", a.getAppointmentDate().toString(),
                        "status", a.getStatus()
                    )).toList());
            result.add(entry);
        }
        return result;
    }

    // ── Appointment queries ───────────────────────────────────────────────────
    public List<Appointment> getAppointmentsByStatus(String status) {
        return appointmentRepository.findByStatus(status);
    }

    public List<Appointment> getAppointmentsByDateRange(LocalDate start, LocalDate end) {
        return appointmentRepository.findByDateRange(start, end);
    }

    public List<Object[]> getAppointmentCountPerDoctor() {
        return appointmentRepository.countAppointmentsPerDoctor();
    }

    public Map<String, Object> getAppointmentsPaginated(int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by("appointmentDate").descending());
        Page<Appointment> pageResult = appointmentRepository.findAll(pageable);

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("appointments", pageResult.getContent());
        response.put("currentPage", pageResult.getNumber());
        response.put("totalItems", pageResult.getTotalElements());
        response.put("totalPages", pageResult.getTotalPages());
        return response;
    }
}
