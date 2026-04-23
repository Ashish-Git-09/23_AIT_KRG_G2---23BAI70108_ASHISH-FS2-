package com.healthhub.repository;

import com.healthhub.entity.Doctor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * DoctorRepository — demonstrates:
 * 1. JPQL queries (@Query with object-level syntax, not SQL table names)
 * 2. Cursor-based pagination (Pageable parameter)
 * 3. JOIN FETCH to solve the N+1 problem
 */
@Repository
public interface DoctorRepository extends JpaRepository<Doctor, Long> {

    // ── JPQL Query 1: Filter by specialization ─────────────────────────────
    // JPQL uses class name 'Doctor' and field name 'specialization', NOT table/column names
    @Query("SELECT d FROM Doctor d WHERE d.specialization = :spec")
    List<Doctor> findBySpecialization(@Param("spec") String specialization);

    // ── JPQL Query 2: Search by name (case-insensitive) ────────────────────
    @Query("SELECT d FROM Doctor d WHERE LOWER(d.name) LIKE LOWER(CONCAT('%', :name, '%'))")
    List<Doctor> searchByName(@Param("name") String name);

    // ── JPQL Query 3: Find experienced doctors ─────────────────────────────
    @Query("SELECT d FROM Doctor d WHERE d.experienceYears >= :years ORDER BY d.experienceYears DESC")
    List<Doctor> findExperiencedDoctors(@Param("years") int minYears);

    // ── Pagination: get page of doctors sorted by name ─────────────────────
    // Pageable parameter enables cursor-based pagination (offset + limit under the hood)
    Page<Doctor> findAll(Pageable pageable);

    // ── JOIN FETCH — solves N+1 problem ────────────────────────────────────
    // WITHOUT this: loading 10 doctors = 1 query for doctors + 10 queries for appointments = 11 queries!
    // WITH JOIN FETCH: 1 single query joins doctors + appointments together
    @Query("SELECT DISTINCT d FROM Doctor d LEFT JOIN FETCH d.appointments")
    List<Doctor> findAllWithAppointments();
}
